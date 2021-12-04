import torch
import torch.nn as nn


class BaseGenerator(nn.Module):
    # generate takes random noise as input, learnable parameter is the input
    # mask. masked input (with noise added) go through the original network
    # and generate masked feature map
    def __init__(self, input_tensor, context, device='cuda:0'):
        super().__init__()
        self.input_tensor = input_tensor
        self.context = context
        self.feature_map = None
        self.device = device

    def init_mean_and_eps(self, *args, **kwargs):
        pass

    def init_input_mask_param(self, input, capacity=None, **kwargs):
        pass

    def forward(self, gaussian):
        pass

    @torch.no_grad()
    def get_feature_map(self):
        pass

    def input_mask(self):
        return torch.sigmoid(self.input_mask_param)

    def clear(self):
        del self.feature_map
        self.feature_map = None
        self.detach()

    def detach(self):
        pass
