from abc import ABCMeta, abstractmethod
from ..bottlenecks import build_feat_iba
from ..gans import build_gan
from ..model_zoo import build_classifiers
from copy import deepcopy
import mmcv


class BaseAttributor(metaclass=ABCMeta):

    def __init__(self,
                 layer: str,
                 classifier: dict,
                 feat_iba: dict,
                 input_iba: dict,
                 gan: dict,
                 use_softmax=True,
                 device='cuda:0'):
        self.device = device
        self.classifier = self.build_classifier(classifier, device=self.device)
        self.layer = layer
        self.use_softmax = use_softmax

        self.feat_iba = build_feat_iba(
            feat_iba, default_args={'context': self})
        self.buffer = {}

        self.input_iba = input_iba
        self.gan = gan

    def build_classifier(self, classifier_cfg, device='cuda:0'):
        classifier = build_classifiers(classifier_cfg).to(device)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False
        return classifier

    def clear_buffer(self):
        self.buffer.clear()

    def estimate(self, data_loader, estimation_cfg):
        self.feat_iba.sigma = None
        self.feat_iba.reset_estimator()
        self.feat_iba.estimate(self.classifier, data_loader, **estimation_cfg)

    def train_gan(self, input_tensor, attr_cfg, logger=None):
        default_args = {
            'input_tensor': input_tensor,
            'context': self,
            # TODO check whether to pass capacity or
            #  sigmoid(self.feat_iba.alpha)
            'feat_mask': self.feat_iba.capacity(),
            'device': self.device
        }
        gan = build_gan(self.gan, default_args=default_args)
        gan.train(logger=logger, **attr_cfg)
        gen_input_mask = gan.generator.input_mask().clone().detach()
        return gen_input_mask

    def make_attribution(self,
                         input_tensor,
                         target,
                         attribution_cfg,
                         logger=None):
        attr_cfg = deepcopy(attribution_cfg)
        if not self.use_softmax:
            feat_iba_batch_size = attr_cfg['feat_iba']['batch_size']
            input_iba_batch_size = attr_cfg['input_iba']['batch_size']
            assert feat_iba_batch_size == input_iba_batch_size, \
                "batch sizes of feat_iba and input_iba should be equal"
        closure = self.get_closure(
            self.classifier,
            target,
            self.use_softmax,
            batch_size=attr_cfg['feat_iba']['batch_size'])
        if logger is None:
            logger = mmcv.get_logger('iba')

        feat_mask = self.train_feat_iba(
            input_tensor, closure, attr_cfg['feat_iba'], logger=logger)

        gen_input_mask = self.train_gan(
            input_tensor, attr_cfg['gan'], logger=logger)

        input_mask = self.train_input_iba(
            input_tensor,
            gen_input_mask,
            closure,
            attr_cfg['input_iba'],
            logger=logger)
        feat_iba_capacity = self.feat_iba.capacity().sum(
            0).clone().detach().cpu().numpy()
        gen_input_mask = gen_input_mask.mean([0, 1]).cpu().numpy()
        self.buffer.update(
            feat_mask=feat_mask,
            input_mask=input_mask,
            gen_input_mask=gen_input_mask,
            feat_iba_capacity=feat_iba_capacity)

    @abstractmethod
    def train_feat_iba(self, input_tensor, closure, attr_cfg, logger=None):
        pass

    @abstractmethod
    def train_input_iba(self,
                        input_tensor,
                        gen_input_mask,
                        closure,
                        attr_cfg,
                        logger=None):
        pass

    @staticmethod
    @abstractmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        pass

    @abstractmethod
    def show_feat_mask(self, **kwargs):
        pass

    def show_gen_input_mask(self, **kwargs):
        pass

    @abstractmethod
    def show_input_mask(self, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def show_mask(mask, **kwargs):
        pass
