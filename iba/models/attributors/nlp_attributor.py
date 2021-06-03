from .base_attributor import BaseAttributor
from .builder import ATTRIBUTORS
from ..bottlenecks import build_input_iba
import torch
import torch.nn.functional as F
import warnings


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
            device=device)

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

    def show_feat_mask(self, **kwargs):
        pass

    def show_gen_input_mask(self, **kwargs):
        pass

    def show_input_mask(self, **kwargs):
        pass

    @staticmethod
    def show_mask(mask, **kwargs):
        pass
