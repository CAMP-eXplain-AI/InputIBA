from mmcv import Registry, build_from_cfg

ATTRIBUTORS = Registry('Attributors')


def build_attributor(cfg, default_args=None):
    return build_from_cfg(cfg, ATTRIBUTORS, default_args=default_args)
