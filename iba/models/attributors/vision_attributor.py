from .base_attributor import BaseAttributor
from .builder import ATTRIBUTORS
from ..bottlenecks import build_input_iba
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import mmcv
from PIL import Image


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
        super(VisionAttributor, self).__init__(layer=layer,
                                               classifier=classifier,
                                               feat_iba=feat_iba,
                                               input_iba=input_iba,
                                               gan=gan,
                                               use_softmax=use_softmax,
                                               device=device)

    def train_feat_iba(self, input_tensor, closure, attr_cfg):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        feat_mask = self.feat_iba.analyze(input_tensor=input_tensor,
                                          model_loss_fn=closure,
                                          **attr_cfg)
        return feat_mask

    def train_input_iba(self, input_tensor, input_iba_cfg, gen_input_mask,
                        closure, attr_cfg):
        assert input_tensor.dim() == 3, \
            f"GAN expect input_tensor to be 3-dimensional, but got a(n) {input_tensor.dim()}d tensor"
        default_args = {
            'input_tensor': input_tensor,
            'input_mask': gen_input_mask
        }
        input_iba = build_input_iba(input_iba_cfg, default_args=default_args)
        input_tensor = input_tensor.unsqueeze(0)
        input_iba_heatmap = input_iba.analyze(input_tensor, closure, **attr_cfg)

        input_mask = torch.sigmoid(input_iba.alpha).detach().cpu().mean(
            [0, 1]).numpy()
        return input_mask

    @staticmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        if use_softmax:
            closure = lambda x: -torch.log_softmax(classifier(x), 1)[:, target
                                                                    ].mean()
        else:
            assert batch_size is not None
            # target is binary encoded and it is for a single sample
            assert isinstance(
                target,
                torch.Tensor) and target.max() <= 1 and target.dim() == 1
            raise NotImplementedError('Currently only support softmax')
        return closure

    def show_feat_mask(self, upscale=False, show=False, out_file=None):
        if not upscale:
            mask = self.buffer['iba_capacity']
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
        VisionAttributor.show_mask(img, show=show, out_file=out_file)

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
