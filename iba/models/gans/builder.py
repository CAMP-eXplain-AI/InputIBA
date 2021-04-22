from mmcv import Registry, build_from_cfg
from .vision_generator import VisionGenerator
from .vision_discriminator import VisionDiscriminator
from .vision_gan import VisionWGAN


GENERATORS = Registry('Generators')
DISCRIMINATORS = Registry('Discriminators')
GANS = Registry('GANs')

GENERATORS.register_module(module=VisionGenerator)

DISCRIMINATORS.register_module(module=VisionDiscriminator)

GANS.register_module(module=VisionWGAN)


def build_generator(cfg, default_args=None):
    return build_from_cfg(cfg, GENERATORS, default_args=default_args)


def build_discriminator(cfg, default_args=None):
    return build_from_cfg(cfg, DISCRIMINATORS, default_args=default_args)


def build_gan(cfg, default_args=None):
    return build_from_cfg(cfg, GANS, default_args=default_args)