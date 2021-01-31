from mmcv import Registry, build_from_cfg
import inspect
import torchvision


DATASETS = Registry('datasets')
PIPELINES = Registry('pipelines')


def register_vision_transforms():
    vision_transforms = []
    for module_name in dir(torchvision.transforms):
        if module_name.startswith('__'):
            continue
        _transform = getattr(torchvision.transforms, module_name)
        if inspect.isclass(_transform):
            PIPELINES.register_module()(_transform)
            vision_transforms.append(module_name)
    return vision_transforms


VISION_TRANSFORMERS = register_vision_transforms()


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args=default_args)