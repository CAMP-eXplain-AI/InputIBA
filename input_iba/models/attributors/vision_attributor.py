import os.path as osp

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch

from ..bottlenecks import build_input_iba
from .base_attributor import BaseAttributor
from .builder import ATTRIBUTORS


@ATTRIBUTORS.register_module()
class VisionAttributor(BaseAttributor):

    def __init__(self,
                 layer: str,
                 classifier: dict,
                 feat_iba: dict,
                 input_iba: dict,
                 gan: dict,
                 use_softmax=True,
                 device='cuda:0'):
        super(VisionAttributor, self).__init__(
            layer=layer,
            classifier=classifier,
            feat_iba=feat_iba,
            input_iba=input_iba,
            gan=gan,
            use_softmax=use_softmax,
            device=device)

    def train_feat_iba(self, input_tensor, closure, attr_cfg, logger=None):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        feat_mask = self.feat_iba.analyze(
            input_tensor=input_tensor,
            model_loss_fn=closure,
            logger=logger,
            **attr_cfg)
        return feat_mask

    def train_input_iba(self,
                        input_tensor,
                        gen_input_mask,
                        closure,
                        attr_cfg,
                        logger=None):
        assert input_tensor.dim() == 3, \
            f"GAN expect input_tensor to be 3-dimensional, but got a(n) " \
            f"{input_tensor.dim()}d tensor"
        default_args = {
            'input_tensor': input_tensor,
            'input_mask': gen_input_mask
        }
        input_iba = build_input_iba(self.input_iba, default_args=default_args)
        input_tensor = input_tensor.unsqueeze(0)
        _ = input_iba.analyze(input_tensor, closure, **attr_cfg, logger=logger)

        input_mask = torch.sigmoid(input_iba.alpha).detach().cpu().mean(
            [0, 1]).numpy()
        return input_mask

    @staticmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        if use_softmax:

            def closure(x):
                loss = -torch.log_softmax(classifier(x), 1)[:, target]
                loss = loss.mean()
                return loss
        else:
            assert batch_size is not None
            # target is binary encoded and it is for a single sample
            assert isinstance(
                target,
                torch.Tensor) and target.max() <= 1 and target.dim() == 1
            raise NotImplementedError('Currently only support softmax')
        return closure

    def show_feat_mask(self, upscale=True, show=False, out_file=None):
        if not upscale:
            mask = self.buffer['feat_iba_capacity']
        else:
            mask = self.buffer['feat_mask']
        mask = mask / mask.max()
        self.show_mask(mask, show=show, out_file=out_file)

    def show_gen_input_mask(self, show=False, out_file=None):
        mask = self.buffer['gen_input_mask']
        self.show_mask(mask, show=show, out_file=out_file)

    def show_input_mask(self, show=False, out_file=None):
        mask = self.buffer['input_mask']
        self.show_mask(mask, show=show, out_file=out_file)

    @staticmethod
    def show_mask(mask, show=False, out_file=None):
        mask_to_show = np.copy(mask)
        if mask.dtype in (float, np.float32, np.float16, np.float128):
            mask = (mask * 255).astype(np.uint8)

        norm = colors.CenteredNorm(0)
        cm = plt.cm.get_cmap('bwr')
        mask_to_show = cm(norm(mask_to_show))
        plt.imshow(mask_to_show, cmap='bwr', norm=colors.CenteredNorm(0))
        plt.axis('off')

        if out_file is not None:
            dir_name = osp.abspath(osp.dirname(out_file))
            mmcv.mkdir_or_exist(dir_name)
            cv2.imwrite(out_file + '.png', mask)
            plt.imsave(out_file + '.JPEG', mask_to_show)
        if not show:
            plt.close()
        else:
            plt.show()
