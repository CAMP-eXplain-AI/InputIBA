import random

import torch
from mmcv import get_logger
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset

from .base_gan import BaseWassersteinGAN, _get_gan_log_string
from .builder import GANS, build_discriminator, build_generator


@GANS.register_module()
class NLPWGAN(BaseWassersteinGAN):

    def __init__(self,
                 generator: dict,
                 discriminator: dict,
                 input_tensor,
                 context,
                 feat_mask=None,
                 device='cuda:0'):
        super(NLPWGAN, self).__init__(
            input_tensor=input_tensor,
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
            'hidden_dim': self.feat_map.shape[-1],
        }
        return build_discriminator(cfg, default_args=default_args)

    def build_data(self, dataset_size, sub_dataset_size, batch_size):
        # create dataset from feature mask and feature map
        num_sub_dataset = int(dataset_size / sub_dataset_size)
        dataset = []
        for idx_subset in range(num_sub_dataset):
            sub_dataset = self.feat_map.expand(-1, sub_dataset_size, -1)
            noise = torch.zeros_like(sub_dataset).normal_()
            std = random.uniform(0, 5)
            mean = random.uniform(-2, 2)
            noise = std * noise + mean
            sub_dataset = self.feat_mask.unsqueeze(1) * sub_dataset + (
                1 - self.feat_mask.unsqueeze(1)) * noise
            dataset.append(sub_dataset)

        dataset = torch.cat(dataset, dim=1)

        # permute dataset because NLP data use dim 1 for batch size
        dataset = dataset.permute(1, 0, 2)
        dataset = dataset.detach()
        tensor_dataset = TensorDataset(dataset)
        dataloader = DataLoader(
            dataset=tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
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
            logger = get_logger('input_iba')
        data_loader = self.build_data(dataset_size, sub_dataset_size,
                                      batch_size)

        # Optimizers
        optimizer_G = RMSprop([{
            "params": self.generator.masker.mean,
            "lr": 0.1
        }, {
            "params": self.generator.masker.eps,
            "lr": 0.05
        }, {
            "params": self.generator.masker.word_embedding_mask_param,
            "lr": 0.03
        }])
        optimizer_D = RMSprop(self.discriminator.parameters(), lr=lr)

        # training
        num_iters = 0
        for epoch in range(epochs):
            for i, data in enumerate(data_loader):

                # train discriminator
                data = data[0]
                data = data.permute(1, 0, 2)
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.input_tensor).float()
                z = z.unsqueeze(-1).unsqueeze(-1).expand(
                    data.shape[0], data.shape[1],
                    100).clone().normal_().to(self.device)
                # z = z.unsqueeze(0).expand(data.shape[0], -1, -1,
                #                           -1).clone().normal_().to(self.device)
                # std = random.uniform(0, 5)
                # mean = random.uniform(-2, 2)
                # noise = std * noise + mean

                # Generate a batch of images
                fake_input = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(data)) + torch.mean(
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
                        log_str = _get_gan_log_string(epoch + 1, epochs,
                                                      num_iters + 1,
                                                      len(data_loader),
                                                      loss_D.item(),
                                                      loss_G.item())
                        logger.info(log_str)
                num_iters += 1

        del data_loader
        self.generator.clear()
