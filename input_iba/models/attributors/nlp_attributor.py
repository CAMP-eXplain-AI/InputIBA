import os.path as osp
import warnings

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from IPython.core.display import HTML, display
from PIL import Image

from ..bottlenecks import build_input_iba
from .base_attributor import BaseAttributor
from .builder import ATTRIBUTORS


@ATTRIBUTORS.register_module()
class NLPAttributor(BaseAttributor):

    def __init__(self,
                 layer: str,
                 classifier: dict,
                 feat_iba: dict,
                 input_iba: dict,
                 gan: dict,
                 use_softmax=True,
                 eval_classifier=False,
                 device='cuda:0'):
        if eval_classifier:
            warnings.warn('Recurrent models need to be in train mode, '
                          'in order to allow gradient backpropagation, '
                          f'but got eval_classifier: {eval_classifier}')
        super(NLPAttributor, self).__init__(
            layer=layer,
            classifier=classifier,
            feat_iba=feat_iba,
            input_iba=input_iba,
            gan=gan,
            use_softmax=use_softmax,
            eval_classifier=False,
            device=device)
        self.text = None

    def set_text(self, text):
        self.text = text

    def train_feat_iba(self, input_tensor, closure, attr_cfg, logger=None):
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(1)

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
        # TODO rewrite to fit NLP
        # assert input_tensor.dim() == 3, \
        #     f"GAN expect input_tensor to be 3-dimensional, but got a(n) " \
        #     f"{input_tensor.dim()}d tensor"
        default_args = {
            'context': self,
            'input_tensor': input_tensor,
            'input_mask': gen_input_mask
        }
        input_iba = build_input_iba(self.input_iba, default_args=default_args)
        input_tensor = input_tensor.unsqueeze(1)
        _ = input_iba.analyze(input_tensor, closure, **attr_cfg, logger=logger)

        input_mask = torch.sigmoid(input_iba.alpha).detach().cpu().mean(
            [1, 2]).numpy()
        return input_mask

    @staticmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        if use_softmax:
            # TODO use_softmax is True, but use BCE loss
            # TODO only for current version, fix this in the future
            # sentence length is part of model's input
            def closure(x):
                text_lengths = torch.tensor([x.shape[0]]).expand(x.shape[1])
                pred = classifier(x, text_lengths)
                return F.binary_cross_entropy_with_logits(pred, target)
        else:
            assert batch_size is not None
            # target is binary encoded and it is for a single sample
            assert isinstance(
                target,
                torch.Tensor) and target.max() <= 1 and target.dim() == 1
            raise NotImplementedError('Currently only support softmax')
        return closure

    def show_feat_mask(self,
                       tokenizer=None,
                       upscale=False,
                       show=False,
                       out_file=None):
        if not upscale:
            mask = self.buffer['feat_iba_capacity']
        else:
            mask = self.buffer['feat_mask']
        self.show_mask(
            mask,
            text=self.text,
            tokenizer=tokenizer,
            show=show,
            out_file=out_file)

    def show_gen_input_mask(self, tokenizer=None, show=False, out_file=None):
        mask = self.buffer['gen_input_mask']
        self.show_mask(
            mask,
            text=self.text,
            tokenizer=tokenizer,
            show=show,
            out_file=out_file)

    def show_input_mask(self, tokenizer=None, show=False, out_file=None):
        mask = self.buffer['input_mask']
        self.show_mask(
            mask,
            text=self.text,
            tokenizer=tokenizer,
            show=show,
            out_file=out_file)

    @staticmethod
    def show_mask(mask, text=None, tokenizer=None, show=False, out_file=None):

        mask = mask / mask.max()
        if show:

            def highlighter(word, word_mask):
                colors = [
                    "#ffffff", "#ffcccc", "#ff9999", '#ff6666', '#ff3333',
                    '#ff0000'
                ]
                if int(word_mask * (len(colors) - 1)) < len(colors):
                    color = colors[int(word_mask * (len(colors) - 1))]
                    word = '<span style="background-color:' \
                           + color + '">' + word + '</span>'
                return word

            highlighted_text = ' '.join([
                highlighter(word, word_mask)
                for (word, word_mask) in zip(tokenizer(text), mask)
            ])

            display(HTML(highlighted_text))

        if out_file is not None:
            mask = (mask * 255).astype(np.uint8)
            mask = np.resize(np.expand_dims(mask, 0), (50, mask.shape[0]))
            dir_name = osp.abspath(osp.dirname(out_file))
            mmcv.mkdir_or_exist(dir_name)
            mask = Image.fromarray(mask, mode='L')
            mask.save(out_file + '.png')
