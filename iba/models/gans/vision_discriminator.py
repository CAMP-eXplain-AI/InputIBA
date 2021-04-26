import torch.nn as nn


class VisionDiscriminator(nn.Module):

    def __init__(self, channels: int, size: tuple):
        super().__init__()
        assert len(
            size
        ) == 2, f"size must be a tuple of (height, width), but got {size}"
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
        self.size = (int(size[0] / 4), int(size[1] / 4))
        # The output of discriminator is no longer a probability,
        # we do not apply sigmoid at the output of discriminator.
        self.output = nn.Sequential(
            nn.Linear(self.channels * self.size[0] * self.size[1], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out = self.main_module(x)
        out_flat = out.view(out.shape[0], -1)
        return self.output(out_flat)
