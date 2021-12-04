from mmcv import Registry, build_from_cfg

from .nlp_feat_iba import NLPFeatureIBA
from .nlp_input_iba import NLPInputIBA
from .vision_feat_iba import VisionFeatureIBA
from .vision_input_iba import VisionInputIBA

FEATURE_IBAS = Registry('Feature IBAs')
INPUT_IBAS = Registry('Input IBAs')

FEATURE_IBAS.register_module(module=VisionFeatureIBA)
INPUT_IBAS.register_module(module=VisionInputIBA)

FEATURE_IBAS.register_module(module=NLPFeatureIBA)
INPUT_IBAS.register_module(module=NLPInputIBA)


def build_feat_iba(cfg, default_args=None):
    return build_from_cfg(cfg, FEATURE_IBAS, default_args=default_args)


def build_input_iba(cfg, default_args=None):
    return build_from_cfg(cfg, INPUT_IBAS, default_args=default_args)
