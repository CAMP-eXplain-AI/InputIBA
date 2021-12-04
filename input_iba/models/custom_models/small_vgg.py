import torch
import torch.nn as nn

from ..model_zoo import MODELS


@MODELS.register_module()
class SmallVGG(nn.Module):
    """
    Smaller VGG model. Compared to VGG-11, this model removes the last two
    convolutional blocks,
    and remove the 4096x4096 hidden linear layer in the classifier.

    Args:
        num_classes (int): number of output classes.
        in_channels (int, optional): input image channels.
        pretrained (str, optional): path to checkpoint. If None, the model
        will be randomly initialized.

    """

    def __init__(self, num_classes, in_channels=3, pretrained=None):
        super(SmallVGG, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes))
        self._initialize_weights(pretrained)

    def _initialize_weights(self, pretrained=None) -> None:
        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
