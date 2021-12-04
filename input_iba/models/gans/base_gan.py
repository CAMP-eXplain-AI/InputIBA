from abc import ABCMeta, abstractmethod

from copy import deepcopy


class BaseWassersteinGAN(metaclass=ABCMeta):

    def __init__(self,
                 generator: dict,
                 discriminator: dict,
                 input_tensor,
                 context,
                 feat_mask=None,
                 device='cuda:0'):
        self.input_tensor = input_tensor
        self.feat_mask = feat_mask
        self.device = device

        generator = deepcopy(generator)
        discriminator = deepcopy(discriminator)

        self.generator = self.build_generator(
            input_tensor=input_tensor, context=context, cfg=generator)
        self.generator = self.generator.to(device)

        self.feat_map = self.get_feat_map()

        self.discriminator = self.build_discriminator(cfg=discriminator)
        self.discriminator = self.discriminator.to(device)

    @abstractmethod
    def build_generator(self, input_tensor, context, cfg):
        pass

    def get_feat_map(self):
        return self.generator.get_feature_map()

    @abstractmethod
    def build_discriminator(self, cfg):
        pass

    @abstractmethod
    def build_data(self, dataset_size, sub_dataset_size, batch_size):
        pass

    @abstractmethod
    def train(self,
              dataset_size=200,
              sub_dataset_size=20,
              lr=5e-5,
              batch_size=32,
              weight_clip=0.1,
              epochs=200,
              critic_iter=5,
              verbose=False,
              logger=None):
        pass


def _get_gan_log_string(epoch, epochs, num_iters, epoch_length, loss_d,
                        loss_g):
    log_str = f'GAN: epoch [{epoch}/{epochs}], '
    log_str += f'[{num_iters % epoch_length}/{epoch_length}], '
    log_str += f'D loss: {loss_d:.5f}, G loss: {loss_g:.5f}'
    return log_str
