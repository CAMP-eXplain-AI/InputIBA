import torch
import torch.nn as nn
import torch.utils.data as Data
from .utils import _to_saliency_map
import random


class Generator(torch.nn.Module):
    # generate takes random noise as input, learnable parameter is the image mask.
    # masked image (with noise added) go through the original network and generate masked feature map
    def __init__(self,
                 model,
                 layer,
                 image,
                 dev='cpu',
                 capacity=None,
                 batch_size=None):
        super().__init__()
        self.image = image
        # use image size
        if capacity is not None:
            image_mask_param = torch.tensor(
                _to_saliency_map(capacity.cpu().detach().numpy(),
                                 image.shape[1:],
                                 data_format="channels_first")).to(dev)
            self.image_mask_param = image_mask_param.expand(
                image.shape[0], -1, -1).clone().unsqueeze(0)
        else:
            self.image_mask_param = torch.zeros(image.shape,
                                                dtype=torch.float).to(dev)
        self.image_mask_param.requires_grad = True
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).to(dev)
        self.mean = torch.tensor([0., 0., 0.]).view(1, -1, 1, 1).to(dev)
        self.mean.requires_grad = True
        # self.eps = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).to(dev)
        self.eps = torch.tensor([1., 1., 1.]).view(1, -1, 1, 1).to(dev)
        self.eps.requires_grad = True
        self.feature_map = None

        self.sigmoid = nn.Sigmoid()

        # register hook in trained classification network
        def store_feature_map(model, input, output):
            self.feature_map = output

        self.model = model.eval()
        # if layer is only accessable in nn module list
        if "[" in layer:
            # read the index
            layer_name = layer.split("[")[0]
            idx_feature = int(layer.split("[")[1].split("]")[0])
            self._hook_handle = getattr(
                self.model,
                str(layer_name))[idx_feature].register_forward_hook(
                    store_feature_map)
        else:
            self._hook_handle = getattr(
                self.model,
                str(layer)).register_forward_hook(store_feature_map)

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
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
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
                 model,
                 layer,
                 image=None,
                 feature_mask=None,
                 feature_noise_mean=None,
                 feature_noise_std=None,
                 dataset_size=800,
                 subdataset_size=50,
                 lr=0.00005,
                 batch_size=32,
                 weight_cliping_limit=0.01,
                 generator_iters=200,
                 critic_iter=5,
                 dev='cpu'):
        # TODO use logging instead of print
        print("WGAN_CP init model.")
        self.image = image
        self.feature_mask = feature_mask
        self.feature_noise_mean = feature_noise_mean
        self.feature_noise_std = feature_noise_std
        self.dataset_size = dataset_size
        self.dev = dev

        self.G = Generator(model,
                           layer,
                           image,
                           dev=self.dev,
                           capacity=feature_mask,
                           batch_size=batch_size)
        self.feature_map = self.G.get_feature_map()

        # channel is determined from feature map
        self.D = Discriminator(self.feature_map.shape[0],
                               self.feature_map.shape[1])

        # create dataset from feature mask and feature map
        self.subdataset_size = subdataset_size
        self.num_subdataset = int(dataset_size / self.subdataset_size)
        dataset = None
        for idx_subset in range(self.num_subdataset):
            sub_dataset = torch.unsqueeze(self.feature_map,
                                          0).expand(self.subdataset_size, -1,
                                                    -1, -1)
            noise = torch.zeros_like(sub_dataset).normal_()
            std = random.uniform(0, 5)
            mean = random.uniform(-2, 2)
            noise = std * noise + mean
            sub_dataset = self.feature_mask * sub_dataset + (
                1 - self.feature_mask) * noise
            if dataset is None:
                dataset = sub_dataset
            else:
                dataset = torch.cat((dataset, sub_dataset))

        dataset = dataset.detach()
        torch_dataset = Data.TensorDataset(dataset)
        self.dataloader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # WGAN values from paper: 0.00005
        self.learning_rate = lr

        self.batch_size = batch_size
        self.weight_cliping_limit = weight_cliping_limit

        self.generator_iters = generator_iters
        self.critic_iter = critic_iter

    def train(self, dev):
        # TODO rename function name
        # Initialize generator and discriminator
        generator = self.G.to(dev)
        discriminator = self.D.to(dev)

        # Optimizers
        optimizer_G = torch.optim.RMSprop([{
            "params": generator.mean,
            "lr": 0.1
        }, {
            "params": generator.eps,
            "lr": 0.05
        }, {
            "params": generator.image_mask_param,
            "lr": 0.003
        }])
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(),
                                          lr=self.learning_rate)

        # ----------
        #  Training
        # ----------

        batches_done = 0
        self.image_mask_history = []
        for epoch in range(self.generator_iters):

            for i, imgs in enumerate(self.dataloader):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                imgs = imgs[0]
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.image)
                z = z.unsqueeze(0).expand(imgs.shape[0], -1, -1,
                                          -1).clone().normal_().to(dev)
                # std = random.uniform(0, 5)
                # mean = random.uniform(-2, 2)
                # noise = std * noise + mean

                # Generate a batch of images
                fake_imgs = generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(imgs)) + torch.mean(
                    discriminator(fake_imgs))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-self.weight_cliping_limit,
                                  self.weight_cliping_limit)

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
                    if epoch % 10 == 0:
                        self.image_mask_history.append(generator.image_mask(
                        ).clone().detach().cpu().numpy())
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.generator_iters,
                           batches_done % len(self.dataloader),
                           len(self.dataloader), loss_D.item(), loss_G.item()))

                # if batches_done % opt.sample_interval == 0:
                #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                batches_done += 1