import torch
from .gan import WGAN_CP
from .pytorch_img_iba import ImageIBA
import matplotlib.pyplot as plt
from ..utils import get_logger
import os.path as osp
import mmcv
import numpy as np
from PIL import Image
from .model_zoo import build_classifiers
from .pytorch import IBA
from copy import deepcopy


class Attributer:
    def __init__(self,
                 cfg: dict,
                 device='cuda:0'):
        self.cfg = deepcopy(cfg)
        self.device = device
        self.classifier = build_classifiers(self.cfg['classifier']).to(self.device)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.layer = self.cfg['layer']
        self.iba = IBA(context=self, **self.cfg['iba'])

        self.buffer = {}

    def clear_buffer(self):
        self.buffer.clear()

    def estimate(self, data_loader, estimation_cfg):
        self.iba.sigma = None
        self.iba.reset_estimate()
        self.iba.estimate(self.classifier,
                          data_loader,
                          device=self.device,
                          **estimation_cfg)

    def train_iba(self, img, closure, attr_cfg):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        iba_heatmap = self.iba.analyze(input_t=img,
                                       model_loss_fn=closure,
                                       **attr_cfg)
        return iba_heatmap

    def train_gan(self, img, attr_cfg, logger=None):
        gan = WGAN_CP(context=self,
                      img=img,
                      feature_mask=self.iba.capacity(),
                      feature_noise_mean=self.iba.estimator.mean(),
                      feature_noise_std=self.iba.estimator.std(),
                      device=self.device)
        gan.train(logger=logger, **attr_cfg)
        gen_img_mask = gan.generator.img_mask().clone().detach()
        return gen_img_mask

    def train_img_iba(self, img_iba_cfg, img, gen_img_mask, closure, attr_cfg):
        img_iba = ImageIBA(
            img=img,
            img_mask=gen_img_mask,
            img_eps_mean=0.0,
            img_eps_std=1.0,
            device=self.device,
            **img_iba_cfg)
        img_iba_heatmap = img_iba.analyze(img.unsqueeze(0), closure, **attr_cfg)
        img_mask = img_iba.sigmoid(img_iba.alpha).detach().cpu().mean([0, 1]).numpy()
        return img_mask, img_iba_heatmap

    def make_attribution(self,
                         img,
                         target,
                         attribution_cfg,
                         logger=None):
        attr_cfg = deepcopy(attribution_cfg)
        closure = lambda x: -torch.log_softmax(
            self.classifier(x), 1)[:, target].mean()
        if logger is None:
            logger = get_logger('iba')

        logger.info('Training Information Bottleneck')
        iba_heatmap = self.train_iba(img, closure, attr_cfg['iba'])

        logger.info('Training GAN')
        gen_img_mask = self.train_gan(img, attr_cfg['gan'], logger=logger)

        logger.info('Training Image Information Bottleneck')
        img_mask, img_iba_heatmap = self.train_img_iba(self.cfg['img_iba'],
                                                       img,
                                                       gen_img_mask,
                                                       closure,
                                                       attr_cfg['img_iba'])

        iba_capacity = self.iba.capacity().sum(0).clone().detach().cpu().numpy()
        gen_img_mask = gen_img_mask.cpu().mean([0, 1]).numpy()
        self.buffer.update(iba_heatmap=iba_heatmap,
                           img_iba_heatmap=img_iba_heatmap,
                           img_mask=img_mask,
                           gen_img_mask=gen_img_mask,
                           iba_capacity=iba_capacity)

    def show_feat_mask(self, upscale=False, show=False, out_file=None):
        if not upscale:
            mask = self.buffer['iba_capacity']
        else:
            mask = self.buffer['iba_heatmap']
        mask = mask / mask.max()
        self._show_mask(mask, show=show, out_file=out_file)

    def show_gen_img_mask(self, show=False, out_file=None):
        mask = self.buffer['gen_img_mask']
        self._show_mask(mask, show=show, out_file=out_file)

    def show_img_mask(self, show=False, out_file=None):
        mask = self.buffer['img_mask']
        self._show_mask(mask, show=show, out_file=out_file)

    @staticmethod
    def _show_mask(mask, show=False, out_file=None):
        plt.imshow(mask)
        plt.axis('off')

        if out_file is not None:
            dir_name = osp.abspath(osp.dirname(out_file))
            mmcv.mkdir_or_exist(dir_name)
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask, mode='L')
            mask.save(out_file + '.png')
            plt.savefig(out_file + '.JPEG', bbox_inches='tight', pad_inches=0)
            if not show:
                plt.close()
        if show:
            plt.show()