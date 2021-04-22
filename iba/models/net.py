import torch
import torch.nn.functional as F
from .gan import WGAN_CP
from .pytorch_img_iba import ImageIBA
import matplotlib.pyplot as plt
import os.path as osp
import mmcv
import numpy as np
from PIL import Image
from .model_zoo import build_classifiers
from .pytorch import IBA
from copy import deepcopy


class Attributor:

    def __init__(self, cfg: dict, device='cuda:0'):
        self.cfg = deepcopy(cfg)
        self.device = device
        self.classifier = build_classifiers(self.cfg['classifier']).to(
            self.device)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.layer = self.cfg['layer']
        use_softmax = cfg.get('use_softmax', True)
        # if not use_softmax:
        #     num_classes = self.cfg['num_classes']
        #     assert num_classes > 1, f"when not using softmax, num_classes should be non-negative, but got {num_classes}"
        #     self.num_classes = num_classes
        self.use_softmax = use_softmax
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
        img_iba = ImageIBA(img=img,
                           img_mask=gen_img_mask,
                           img_eps_mean=0.0,
                           img_eps_std=1.0,
                           device=self.device,
                           **img_iba_cfg)
        img_iba_heatmap = img_iba.analyze(img.unsqueeze(0), closure, **attr_cfg)
        img_mask = img_iba.sigmoid(img_iba.alpha).detach().cpu().mean([0, 1]).numpy()
        return img_mask, img_iba_heatmap

    @staticmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        if use_softmax:
            closure = lambda x: -torch.log_softmax(classifier(x), 1)[:, target].mean()
        else:
            assert batch_size is not None
            # target is binary encoded and it is for a single sample
            assert isinstance(target, torch.Tensor) and target.max() <= 1 and target.dim() == 1
            raise NotImplementedError('Currently only support softmax')
        return closure

    def make_attribution(self, img, target, attribution_cfg, logger=None):
        attr_cfg = deepcopy(attribution_cfg)
        if not self.use_softmax:
            assert attr_cfg['iba']['batch_size'] == attr_cfg['img_iba']['batch_size'], \
                "batch sizes of iba and img_iba should be equal"
        closure = self.get_closure(self.classifier,
                                   target,
                                   self.use_softmax,
                                   batch_size=attr_cfg['iba']['batch_size'])
        if logger is None:
            logger = mmcv.get_logger('iba')

        logger.info('Training Information Bottleneck')

        # # feature mask at image size (with upscale)
        iba_heatmap = self.train_iba(img, closure, attr_cfg['iba'])

        logger.info('Training GAN')
        # get generated image mask (size of image size)
        gen_img_mask = self.train_gan(img, attr_cfg['gan'], logger=logger)

        logger.info('Training Image Information Bottleneck')
        img_mask, img_iba_heatmap = self.train_img_iba(self.cfg['img_iba'], img,
                                                       gen_img_mask, closure,
                                                       attr_cfg['img_iba'])

        # feature mask at feature size (without upscale)
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
        self.show_mask(mask, show=show, out_file=out_file)

    def show_gen_img_mask(self, show=False, out_file=None):
        mask = self.buffer['gen_img_mask']
        self.show_mask(mask, show=show, out_file=out_file)

    def show_img_mask(self, show=False, out_file=None):
        mask = self.buffer['img_mask']
        self.show_mask(mask, show=show, out_file=out_file)

    @staticmethod
    def show_img(img,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 show=False,
                 out_file=None):
        if isinstance(img, torch.Tensor):
            assert img.max().item() < 1
            mean = torch.tensor(mean).to(img)
            std = torch.tensor(std).to(img)
        else:
            assert img.max() < 1
            mean = np.array(mean)
            std = np.array(std)
        img = img * std + mean
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        Attributor.show_mask(img, show=show, out_file=out_file)

    @staticmethod
    def show_mask(mask, show=False, out_file=None):
        if mask.dtype in (float, np.float32, np.float16, np.float128):
            assert mask.max() <= 1.0
            mask = (mask * 255).astype(np.uint8)
        plt.imshow(mask)
        plt.axis('off')

        if out_file is not None:
            dir_name = osp.abspath(osp.dirname(out_file))
            mmcv.mkdir_or_exist(dir_name)
            mask = Image.fromarray(mask, mode='L')
            mask.save(out_file + '.png')
            plt.savefig(out_file + '.JPEG', bbox_inches='tight', pad_inches=0)
            if not show:
                plt.close()
        if show:
            plt.show()
