import os
from torchvision import models
from .custom_models.deep_lstm import DeepLSTM
import torch
from copy import deepcopy
from mmcv import Registry, build_from_cfg

MODELS = Registry('models')


def build_classifiers(cfg, default_args=None):
    cfg = deepcopy(cfg)
    assert 'source' in cfg, \
        "source is not specified, it can be one of " \
        "('custom', 'torchvision','timm')"

    source = cfg.pop('source')
    assert source in ('torchvision', 'custom', 'timm', 'nlp'), \
        f"source should be on of ('custom', 'torchvision', 'timm', 'nlp'), but " \
        f"got {source}"

    if source == 'torchvision':
        return build_torchvision_classifiers(cfg)
    elif source == 'timm':
        return build_timm_classifiers(cfg)
    elif source == 'nlp':
        return build_lstm_classifiers(cfg)
    else:
        return build_from_cfg(cfg, MODELS, default_args=default_args)


def build_torchvision_classifiers(cfg):
    cfg = deepcopy(cfg)
    model_name = cfg.pop('type')
    pretrained = cfg.pop('pretrained', True)
    assert isinstance(pretrained, (bool, str))
    _builder = getattr(models, model_name)
    # if pretrained is a path, first build a randomly initialized model,
    # then load the pretrained weight; if pretrained is a bool, just call the
    # torchvision's builder, and pass the boolean to the builder
    if isinstance(pretrained, str):
        cfg.update({"pretrained": False})
        model = _builder(**cfg)
        ckpt = torch.load(pretrained)
        model.load_state_dict(ckpt)
    else:
        cfg.update({'pretrained': pretrained})
        model = _builder(**cfg)
    return model


def build_lstm_classifiers(cfg):
    INPUT_DIM = 25002
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 1
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    PAD_IDX = 1

    model = DeepLSTM(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)

    # select a model to analyse
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), cfg.pop("pretrained"))))
    return model


def build_timm_classifiers(cfg):
    import timm
    cfg = deepcopy(cfg)
    model_name = cfg.pop('type')
    return timm.create_model(model_name, **cfg)


def get_module(model, module):
    r"""Returns a specific layer in a model based.
    Shameless copy from `TorchRay
    <https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/common.py>`_
    :attr:`module` is either the name of a module (as given by the
    :func:`named_modules` function for :class:`torch.nn.Module` objects) or
    a :class:`torch.nn.Module` object. If :attr:`module` is a
    :class:`torch.nn.Module` object, then :attr:`module` is returned unchanged.
    If :attr:`module` is a str, the function searches for a module with the
    name :attr:`module` and returns a :class:`torch.nn.Module` if found;
    otherwise, ``None`` is returned.
    Args:
        model (:class:`torch.nn.Module`): model in which to search for layer.
        module (str or :class:`torch.nn.Module`): name of layer (str) or the
            layer itself (:class:`torch.nn.Module`).
    Returns:
        :class:`torch.nn.Module`: specific PyTorch layer (``None`` if the layer
            isn't found).
    """
    if isinstance(module, torch.nn.Module):
        return module

    assert isinstance(module, str)
    if module == '':
        return model

    for name, curr_module in model.named_modules():
        if name == module:
            return curr_module

    return None
