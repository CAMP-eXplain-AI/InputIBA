import albumentations as A
import inspect
from albumentations.pytorch import ToTensorV2
from mmcv import Registry, build_from_cfg
from typing import Dict, List, Union

DATASETS = Registry('datasets')
PIPELINES = Registry('pipelines')


def register_albu_transforms():
    albu_transforms = []
    for module_name in dir(A):
        if module_name.startswith('_'):
            continue
        transform = getattr(A, module_name)
        if inspect.isclass(transform) and issubclass(transform,
                                                     A.BasicTransform):
            PIPELINES.register_module()(transform)
            albu_transforms.append(module_name)
    return albu_transforms


albu_transforms = register_albu_transforms()
PIPELINES.register_module(module=ToTensorV2)


def build_pipeline(cfg: Union[Dict, List], default_args=None):
    if isinstance(cfg, Dict):
        return build_from_cfg(cfg, PIPELINES)
    else:
        pipeline = []
        for transform_cfg in cfg:
            t = build_pipeline(transform_cfg)
            pipeline.append(t)
        if default_args is None:
            default_args = {}
        return A.Compose(pipeline, **default_args)


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args=default_args)
