from iba.models.net import Attributer
import torch
import numpy as np
from scipy import signal
from copy import deepcopy
from iba.models import get_module
from .base import BaseEvaluation
from ..utils import get_logger


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(m.bias)


def perturb_model(model, layers):
    for layer in layers:
        module = get_module(model, layer)
        module.apply(weights_init)


class SanityCheck(BaseEvaluation):
    def __init__(self,
                 attributer: Attributer):
        self.attributer = attributer
        self.ori_state_dict = deepcopy(self.attributer.classifier.state_dict())
        self.model_layers = self.filter_names([n[0] for n in self.attributer.classifier.named_modules()])
        self.logger = get_logger('iba')

    def reload(self):
        self.logger.info('Reload state dict')
        self.attributer.classifier.load_state_dict(self.ori_state_dict)
        self.attributer.classifier.to(self.attributer.device)
        self.attributer.classifier.eval()
        for p in self.attributer.classifier.parameters():
            p.requires_grad = False

    def evaluate(self, heatmap, img, target, attribution_cfg, perturb_layers, check='gan'): # noqa
        assert check in ['gan', 'img_iba'], f"check must be one of 'gan' or 'img_iba', but got {check}"
        attr_cfg = deepcopy(attribution_cfg)
        model_layers = deepcopy(self.model_layers)
        # start from the last layer
        model_layers = model_layers[::-1]
        ssim_all = []
        for l in perturb_layers:
            # reload state_dict
            self.reload()
            self.logger.info(f'Perturb {l} and subsequent layers')
            p_layers = []
            for m in model_layers:
                if l != m:
                    p_layers.append(m)
                else:
                    break
            p_layers.append(l)
            self.logger.info(f"Following layers will be perturbed: [{', '.join(p_layers)}]")
            ssim_val = self.sanity_check_single(img=img,
                                                target=target,
                                                attr_cfg=attr_cfg,
                                                perturb_layers=p_layers,
                                                ori_img_mask=heatmap,
                                                check=check)
            ssim_all.append(ssim_val)
        return dict(ssim_all=ssim_all)

    def sanity_check_single(self,
                            img,
                            target,
                            attr_cfg,
                            perturb_layers,
                            ori_img_mask,
                            check='gan'):
        closure = lambda x: -torch.log_softmax(
            self.attributer.classifier(x), 1)[:, target].mean()

        _ = self.attributer.train_iba(img, closure, attr_cfg['iba'])
        if check == 'gan':
            perturb_model(self.attributer.classifier, perturb_layers)
            gen_img_mask = self.attributer.train_gan(img, attr_cfg['gan'])
        else:
            gen_img_mask = self.attributer.train_gan(img, attr_cfg['gan'])
            perturb_model(self.attributer.classifier, perturb_layers)
        img_mask, _ = self.attributer.train_img_iba(self.attributer.cfg['img_iba'],
                                                    img,
                                                    gen_img_mask=gen_img_mask,
                                                    closure=closure,
                                                    attr_cfg=attr_cfg['img_iba'])
        return self.ssim(ori_img_mask, img_mask)

    @staticmethod
    def ssim(mask_1, mask_2):
        mask_1 = SanityCheck.convert_mask(mask_1)
        mask_2 = SanityCheck.convert_mask(mask_2)
        return _ssim(mask_1, mask_2).mean()

    @staticmethod
    def convert_mask(m):
        if m.dtype == float:
            assert m.max() <= 1.0
            m = (m * 255).astype(np.uint8)
        return m

    @staticmethod
    def filter_names(names):
        res = []
        for i in range(len(names) - 1):
            if not names[i] in names[i + 1]:
                res.append(names[i])
        res.append(names[-1])
        return res


def fspecial_gauss(size, sigma):
    """Shameless copy from https://github.com/mubeta06/python/blob/master/signal_processing/sp/gauss.py"""
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def _ssim(img1, img2, cs_map=False):
    """Shameless copy from https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py.
    Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))