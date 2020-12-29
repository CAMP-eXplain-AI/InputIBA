import torch
import torch.nn as nn
import copy
import torch.utils.data as Data

class Generator(torch.nn.Module):
    # generate takes random noise as input, learnable parameter is the image mask.
    # masked image (with noise added) go through the original network and generate masked feature map
    def __init__(self, model, layer, image):
        super().__init__()
        self.image = image
        # use image size
        self.image_mask_param = torch.randn(image.shape, dtype=torch.float, requires_grad=True)
        self.mean = torch.randn(image.shape, dtype=torch.float, requires_grad=True)
        self.eps = torch.randn(image.shape, dtype=torch.float, requires_grad=True)
        self.feature_map = None

        self.sigmoid = nn.Sigmoid()
        # register hook in trained classification network
        def store_feature_map(model, input, output):
            self.feature_map = output

        self.model = model
        self._hook_handle = getattr(self.model, str(layer)).register_forward_hook(store_feature_map)

    def forward(self, gaussian):
        noise = self.eps * gaussian + self.mean
        image_mask = self.sigmoid(self.image_mask_param)
        masked_image = image_mask * self.image + (1 - image_mask) * noise
        _ = self.model(masked_image)
        masked_feature_map = self.feature_map
        return masked_feature_map

    def get_feature_map(self):
        _ = self.model(self.image.unsqueeze(0))
        return self.feature_map.squeeze(0)

    # function for retrieve image mask
    def image_mask(self):
        return self.sigmoid(self.image_mask_param)


class Discriminator(torch.nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (CxFeatureMapSizexFeatureMapSize)
        # Output_dim = 1
        # self.main_module = nn.Sequential(
        #     # Image (Cx32x32)
        #     nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # State (256x16x16)
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # State (512x8x8)
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.LeakyReLU(0.2, inplace=True))
        # # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            # nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))
            nn.Linear(int(channels*size*size), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # x = self.main_module(x)
        print(x.shape)
        x_flat = x.view(x.shape[0], -1)
        print(x_flat.shape)
        return self.output(x_flat)


class WGAN_CP(object):
    def __init__(self, model, layer,
                 image=None, feature_mask=None,
                 feature_noise_mean=None, feature_noise_std=None,
                 dataset_size=300, lr=0.00005, batch_size=32, weight_cliping_limit=0.01,
                 generator_iters=10, critic_iter=5):
        print("WGAN_CP init model.")
        self.image = image
        self.feature_mask = feature_mask
        self.feature_noise_mean = feature_noise_mean
        self.feature_noise_std = feature_noise_std
        self.dataset_size = dataset_size

        self.G = Generator(model, layer, image)
        self.feature_map = self.G.get_feature_map()

        # channel is determined from feature map
        self.D = Discriminator(self.feature_map.shape[0], self.feature_map.shape[1])

        # create dataset from feature mask and feature map
        dataset = torch.unsqueeze(self.feature_map, 0).expand(self.dataset_size, -1, -1, -1)
        noise = torch.zeros_like(dataset).normal_()
        noise = self.feature_noise_std * noise + self.feature_noise_mean
        dataset = self.feature_mask * dataset + (1 - self.feature_mask) * noise
        dataset = dataset.detach()
        torch_dataset = Data.TensorDataset(dataset)
        self.dataloader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        # WGAN values from paper: 0.00005
        self.learning_rate = lr

        self.batch_size = batch_size
        self.weight_cliping_limit = weight_cliping_limit

        self.generator_iters = generator_iters
        self.critic_iter = critic_iter

    def train(self, dev):
        # Initialize generator and discriminator
        generator = self.G.to(dev)
        discriminator = self.D.to(dev)


        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=self.learning_rate)

        # ----------
        #  Training
        # ----------

        batches_done = 0
        for epoch in range(self.generator_iters):

            for i, imgs in enumerate(self.dataloader):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                imgs = imgs[0]
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.image)
                z = z.unsqueeze(0).expand(imgs.shape[0], -1, -1, -1).normal_().to(dev)

                # Generate a batch of images
                fake_imgs = generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(imgs)) + torch.mean(discriminator(fake_imgs))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                # Train the generator every n_critic iterations
                if i % self.critic_iter == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(discriminator(gen_imgs))

                    loss_G.backward()
                    optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.generator_iters, batches_done % len(self.dataloader), len(self.dataloader), loss_D.item(),
                           loss_G.item())
                    )

                # if batches_done % opt.sample_interval == 0:
                #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                batches_done += 1