import torch
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader
from .base_gan import BaseWassersteinGAN
from .builder import build_generator, build_discriminator, GANS
from mmcv import get_logger
import random


@GANS.register_module()
class VisionWGAN(BaseWassersteinGAN):

    def __init__(self,
                 generator: dict,
                 discriminator: dict,
                 input_tensor,
                 context,
                 feat_mask=None,
                 device='cuda:0'):
        super(VisionWGAN, self).__init__(input_tensor=input_tensor,
                                         context=context,
                                         generator=generator,
                                         discriminator=discriminator,
                                         feat_mask=feat_mask,
                                         device=device)

    def build_generator(self, input_tensor, context, cfg):
        default_args = {
            'input_tensor': self.input_tensor,
            'context': context,
            'device': self.device,
            'capacity': self.feat_mask
        }
        return build_generator(cfg, default_args=default_args)

    def build_discriminator(self, cfg):
        default_args = {
            'channels': self.feat_map.shape[0],
            'size': self.feat_map.shape[1:]
        }
        return build_discriminator(cfg, default_args=default_args)

    def build_data(self, dataset_size, sub_dataset_size, batch_size):
        # create dataset from feature mask and feature map
        num_sub_dataset = int(dataset_size / sub_dataset_size)
        dataset = []
        for idx_subset in range(num_sub_dataset):
            sub_dataset = self.feat_map.unsqueeze(0).expand(
                sub_dataset_size, -1, -1, -1)
            noise = torch.zeros_like(sub_dataset).normal_()
            std = random.uniform(0, 5)
            mean = random.uniform(-2, 2)
            noise = std * noise + mean
            sub_dataset = self.feat_mask * sub_dataset + (
                1 - self.feat_mask) * noise
            dataset.append(sub_dataset)

        dataset = torch.cat(dataset, dim=0)
        dataset = dataset.detach()
        tensor_dataset = TensorDataset(dataset)
        dataloader = DataLoader(dataset=tensor_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0)
        return dataloader

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
        # TODO add learning rate scheduler
        # Initialize generator and discriminator
        if logger is None:
            logger = get_logger('iba')
        data_loader = self.build_data(dataset_size, sub_dataset_size,
                                      batch_size)

        # Optimizers
        optimizer_G = RMSprop([{
            "params": self.generator.mean,
            "lr": 0.1
        }, {
            "params": self.generator.eps,
            "lr": 0.05
        }, {
            "params": self.generator.input_mask_param,
            "lr": 0.003
        }])
        optimizer_D = RMSprop(self.discriminator.parameters(), lr=lr)

        # training
        batches_done = 0
        for epoch in range(epochs):
            for i, data in enumerate(data_loader):

                # train discriminator
                # data is a tuple of one element
                real_input = data[0]
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.input_tensor)
                z = z.unsqueeze(0).expand(real_input.shape[0], -1, -1,
                                          -1).clone().normal_().to(self.device)

                # Generate a batch of images
                fake_input = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(
                    self.discriminator(real_input)) + torch.mean(
                        self.discriminator(fake_input))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

                # Train the generator every n_critic iterations
                if i % critic_iter == 0:
                    # train generator
                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_input = self.generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_input))

                    loss_G.backward()
                    optimizer_G.step()
                    if verbose:
                        log_str = f'GAN: epoch [{epoch + 1}/{epochs}], '
                        log_str += f'[{batches_done % len(data_loader)}/{len(data_loader)}], '
                        log_str += f'D loss: {loss_D.item():.5f}, G loss: {loss_G.item():.5f}'
                        logger.info(log_str)
                batches_done += 1

        del data_loader
        self.generator.clear()
