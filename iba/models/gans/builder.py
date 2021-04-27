from mmcv import Registry, build_from_cfg

GENERATORS = Registry('Generators')
DISCRIMINATORS = Registry('Discriminators')
GANS = Registry('GANs')


def build_generator(cfg, default_args=None):
    return build_from_cfg(cfg, GENERATORS, default_args=default_args)


def build_discriminator(cfg, default_args=None):
    return build_from_cfg(cfg, DISCRIMINATORS, default_args=default_args)


def build_gan(cfg, default_args=None):
    return build_from_cfg(cfg, GANS, default_args=default_args)
