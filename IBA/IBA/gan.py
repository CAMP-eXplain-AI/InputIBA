import torch
import torch.nn as nn
import copy
from torch.autograd import Variable


class Generator(torch.nn.Module):
    # generate takes random noise as input, learnable parameter is the image mask.
    # masked image (with noise added) go through the original network and generate masked feature map
    def __init__(self, model, layer, image):
        super().__init__()
        self.image = image
        # TODO use image size
        self.image_mask_param = torch.randn((D, H), dtype=torch.FloatTensor, requires_grad=True)
        self.mean = torch.randn((D, H), dtype=torch.FloatTensor, requires_grad=True)
        self.eps = torch.randn((D, H), dtype=torch.FloatTensor, requires_grad=True)
        self.feature_map = None

        self.sigmoid = nn.Sigmoid()
        # register hook in trained classification network
        def store_feature_map(model, input, output):
            self.feature_map = output
        self.model = copy.deepcopy(model)
        self._hook_handle = getattr(self.model, str(layer)).register_forward_hook(store_feature_map)

    def forward(self, gaussian):
        # gaussian = self.image_mask_param.data.new(self.image_mask_param.size()).normal_()
        noise = self.eps * gaussian + self.mean
        image_mask = self.sigmoid(self.image_mask_param)
        masked_image = image_mask * self.image + (1 - image_mask) * noise
        _ = self.model(masked_image)
        masked_feature_map = self.feature_map
        return masked_feature_map

    # function for retrieve image mask
    def image_mask(self):
        return self.sigmoid(self.image_mask_param)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (CxFeatureMapSizexFeatureMapSize)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class WGAN_CP(object):
    def __init__(self, args):
        print("WGAN_CP init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # WGAN values from paper
        self.learning_rate = 0.00005

        self.batch_size = 64
        self.weight_cliping_limit = 0.01

        self.generator_iters = args.generator_iters
        self.critic_iter = 5

    def train(self, train_loader, dev):
        # Initialize generator and discriminator
        generator = self.G.to(dev)
        discriminator = self.D.to(dev)


        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=self.learning_rate)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        batches_done = 0
        for epoch in range(self.generator_iters):

            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = self.image_mask_param.data.new(self.image_mask_param.size()).normal_().to(dev)

                # Generate a batch of images
                fake_imgs = generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

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
                        % (epoch, n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(),
                           loss_G.item())
                    )

                if batches_done % opt.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                batches_done += 1