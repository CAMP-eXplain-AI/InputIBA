from abc import ABCMeta, abstractmethod
from ..bottlenecks import build_feat_iba
from ..gans import build_gan
from ..model_zoo import build_classifiers
from copy import deepcopy
import mmcv


class BaseAttributor(abc=ABCMeta):

    def __init__(self, cfg: dict, device='cuda:0'):
        self.cfg = deepcopy(cfg)
        self.device = device
        self.classfier = self.build_classifier(self.cfg['classifier'], device=self.device)
        self.layer = self.cfg['layer']
        use_softmax = cfg.get('use_softmax', True)
        self.use_softmax = use_softmax

        self.iba = build_feat_iba(self.cfg['iba'],
                                  default_args={'context': self})
        self.buffer = {}

    def build_classifier(self, classifier_cfg, device='cuda:0'):
        classifier = build_classifiers(classifier_cfg).to(device)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False
        return classifier

    def clear_buffer(self):
        self.buffer.clear()

    def estimate(self, data_loader, estimation_cfg):
        self.iba.sigma = None
        self.iba.reset_estimate()
        self.iba.estimate(self.classfier,
                          data_loader,
                          device=self.device,
                          **estimation_cfg)

    @abstractmethod
    def train_iba(self, input_tensor, closure, attr_cfg):
        pass

    def train_gan(self, input_tensor, attr_cfg, logger=None):
        default_args = {'input_tensor': input_tensor,
                        'context': self,
                        'feat_mask': self.iba.capacity(),
                        'device': self.device}
        gan = build_gan(self.cfg['gan'], default_args=default_args)
        gan.train(logger=logger, **attr_cfg)
        gen_input_mask = gan.generator.input_mask().clone().detach()
        return gen_input_mask

    @abstractmethod
    def train_input_iba(self,
                        input_tensor,
                        input_iba_cfg,
                        gen_input_mask,
                        closure,
                        attr_cfg):
        pass

    @abstractmethod
    @staticmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        pass

    def make_attribution(self, input_tensor, target, attribution_cfg, logger=None):
        attr_cfg = deepcopy(attribution_cfg)
        if not self.use_softmax:
            assert attr_cfg['iba']['batch_size'] == attr_cfg['img_iba']['batch_size'], \
                "batch sizes of iba and img_iba should be equal"
        closure = self.get_closure(self.classfier,
                                   target,
                                   self.use_softmax,
                                   batch_size=attr_cfg['iba']['batch_size'])
        if logger is None:
            logger = mmcv.get_logger('iba')

        logger.info('Training Feature Information Bottleneck')
        iba_heatmap = self.train_iba(input_tensor, closure, attr_cfg['iba'])

        logger.info('Training GAN')
        gen_input_mask = self.train_gan(input_tensor, attr_cfg['gan'], logger=logger)

        logger.info('Training Input Information Bottleneck')
        # TODO rename `input_iba_heatmap`
        input_mask, input_iba_heatmap = self.train_input_iba(input_tensor,
                                                             self.cfg['input_iba'],
                                                             gen_input_mask,
                                                             closure,
                                                             attr_cfg['input_iba'])
        iba_capacity = self.iba.capacity().sum(0).clone().detach().cpu().numpy()
        gen_input_mask = gen_input_mask.mean([0, 1]).numpy()
        self.buffer.update(iba_heatmap=iba_heatmap,
                           input_iba_heatmap=input_iba_heatmap,
                           input_mask=input_mask,
                           gen_input_mask=gen_input_mask,
                           iba_capacity=iba_capacity)

    @abstractmethod
    def show_feat_mask(self, **kwargs):
        pass

    def show_gen_input_mask(self, **kwargs):
        pass

    @abstractmethod
    def show_input_mask(self, **kwargs):
        pass

    @abstractmethod
    @staticmethod
    def show_mask(self, mask, **kwargs):
        pass
