import torch
import torch.nn as nn
import torch.utils.data as Data
from .utils import _to_saliency_map
import random
from ..utils import get_logger
from .model_zoo import get_module


class Generator(torch.nn.Module):
    # generate takes random noise as input, learnable parameter is the img mask.
    # masked img (with noise added) go through the original network and generate masked feature map
    def __init__(self,
                 img,
                 context=None,
                 layer=None,
                 device='cuda:0',
                 capacity=None):
        super().__init__()
        self.img = img

        assert (context is None) ^ (layer is None)
        self.context = context
        self.layer = layer

        # use img size
        # TODO make image_mask_param a Parameter
        if capacity is not None:
            image_mask_param = torch.tensor(
                _to_saliency_map(capacity.cpu().detach().numpy(),
                                 img.shape[1:],
                                 data_format="channels_first")).to(device)
            self.image_mask_param = image_mask_param.expand(
                img.shape[0], -1, -1).clone().unsqueeze(0)
        else:
            self.image_mask_param = torch.zeros(img.shape,
                                                dtype=torch.float).to(device)
        self.image_mask_param.requires_grad = True
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).to(dev)
        # TODO make mean and eps Parameters.
        self.mean = torch.tensor([0., 0., 0.]).view(1, -1, 1, 1).to(device)
        self.mean.requires_grad = True
        # self.eps = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).to(dev)
        self.eps = torch.tensor([1., 1., 1.]).view(1, -1, 1, 1).to(device)
        self.eps.requires_grad = True
        self.feature_map = None

        self.sigmoid = nn.Sigmoid()

        # TODO use out of place function to avoid setting attributes
        # register hook in trained classification network
        def store_feature_map(model, input, output):
            self.feature_map = output

        if self.context is not None:
            self._hook_handle = get_module(self.context.classifier, self.context.layer
                                           ).register_forward_hook(store_feature_map)
        elif self.layer is not None:
            self._hook_handle = self.layer.register_forward_hook(store_feature_map)
        else:
            raise ValueError('context and layer cannot be Non at the same time')

    def forward(self, gaussian):
        noise = self.eps * gaussian + self.mean
        image_mask = self.sigmoid(self.image_mask_param)
        masked_image = image_mask * self.img + (1 - image_mask) * noise
        _ = self.model(masked_image)
        masked_feature_map = self.feature_map
        return masked_feature_map

    @torch.no_grad()
    def get_feature_map(self):
        _ = self.model(self.img.unsqueeze(0))
        return self.feature_map.squeeze(0)

    def image_mask(self):
        return self.sigmoid(self.image_mask_param)

    def clear(self):
        del self.feature_map
        self.feature_map = None
        self.detach()

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        else:
            raise ValueError(
                "Cannot detach hock. Either you never attached or already detached.")


class Discriminator(torch.nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (CxFeatureMapSizexFeatureMapSize)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels,
                      out_channels=channels * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=channels * 2,
                      out_channels=channels * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=channels * 4,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.LeakyReLU(0.2, inplace=True))
        # # output of main module --> State (1024x4x4)

        # channels and sizes for fully connected layer
        self.channels = channels
        self.size = int(size / 4)
        self.output = nn.Sequential(
            # The output of discriminator is no longer a probability, we do not apply sigmoid at the output of discriminator.
            # nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))
            nn.Linear(self.channels * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out = self.main_module(x)
        out_flat = out.view(out.shape[0], -1)
        return self.output(out_flat)


class WGAN_CP(object):
    def __init__(self,
                 context=None,
                 layer=None,
                 img=None,
                 feature_mask=None,
                 feature_noise_mean=None,
                 feature_noise_std=None,
                 device='cuda:0'):
        print("WGAN_CP init model.")
        self.img = img
        self.feature_mask = feature_mask
        self.feature_noise_mean = feature_noise_mean
        self.feature_noise_std = feature_noise_std
        self.device = device

        self.generator = Generator(img=img,
                                   context=context,
                                   layer=layer,
                                   device=self.device,
                                   capacity=feature_mask).to(self.device)
        self.feature_map = self.generator.get_feature_map()

        # channel is determined from feature map
        self.discriminator = Discriminator(self.feature_map.shape[0],
                                           self.feature_map.shape[1]).to(self.device)

    def _build_data(self, dataset_size, sub_dataset_size, batch_size):
        # create dataset from feature mask and feature map
        num_sub_dataset = int(dataset_size / sub_dataset_size)
        dataset = []
        for idx_subset in range(num_sub_dataset):
            sub_dataset = self.feature_map.unsqueeze(0).expand(sub_dataset_size, -1, -1, -1)
            noise = torch.zeros_like(sub_dataset).normal_()
            std = random.uniform(0, 5)
            mean = random.uniform(-2, 2)
            noise = std * noise + mean
            sub_dataset = self.feature_mask * sub_dataset + (
                    1 - self.feature_mask) * noise
            dataset.append(sub_dataset)

        dataset = torch.cat(dataset, dim=0)
        dataset = dataset.detach()
        tensor_dataset = Data.TensorDataset(dataset)
        dataloader = Data.DataLoader(
            dataset=tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        return dataloader

    def train(self,
              logger=None,
              dataset_size=200,
              sub_dataset_size=20,
              lr=0.00005,
              batch_size=32,
              weight_clip=0.01,
              epochs=200,
              critic_iter=5):
        # TODO add learning rate scheduler
        # Initialize generator and discriminator
        if logger is None:
            logger = get_logger('iba')
        data_loader = self._build_data(dataset_size, sub_dataset_size, batch_size)

        # Optimizers
        optimizer_G = torch.optim.RMSprop([{
            "params": self.generator.mean,
            "lr": 0.1
        }, {
            "params": self.generator.eps,
            "lr": 0.05
        }, {
            "params": self.generator.image_mask_param,
            "lr": 0.003
        }])
        optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)

        # training
        batches_done = 0
        for epoch in range(epochs):
            for i, imgs in enumerate(data_loader):

                # train discriminator
                imgs = imgs[0]
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.img)
                z = z.unsqueeze(0).expand(imgs.shape[0], -1, -1,
                                          -1).clone().normal_().to(self.device)
                # std = random.uniform(0, 5)
                # mean = random.uniform(-2, 2)
                # noise = std * noise + mean

                # Generate a batch of images
                fake_imgs = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(imgs)) + torch.mean(
                    self.discriminator(fake_imgs))

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
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_imgs))

                    loss_G.backward()
                    optimizer_G.step()
                    log_str = f'[Epoch{epoch + 1}/{epochs}], '
                    log_str += f'[{batches_done % len(data_loader)}/{len(data_loader)}], '
                    log_str += f'D loss: {loss_D.item():.5f}, G loss: {loss_G.item():.5f}'
                    logger.info(log_str)
                batches_done += 1

        del data_loader
        self.generator.clear()