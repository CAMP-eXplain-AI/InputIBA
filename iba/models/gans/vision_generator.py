import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import _to_saliency_map
from ..model_zoo import get_module
from .base_generator import BaseGenerator


class VisionGenerator(BaseGenerator):
    # generate takes random noise as input, learnable parameter is the img mask.
    # masked img (with noise added) go through the original network and generate masked feature map
    def __init__(self, input_tensor, context, device='cuda:0', capacity=None):
        super().__init__(input_tensor, context, device=device)
        self.input_mask_param = self.init_input_mask_param(
            input_tensor, capacity)
        self.mean, self.eps = self.init_mean_and_eps()

        # register hook in trained classification network
        def store_feature_map(model, input, output):
            self.feature_map = output

        module = get_module(self.context.classifier, self.context.layer)
        self._hook_handle = module.register_forward_hook(store_feature_map)

    def init_input_mask_param(self, input_tensor, capacity=None):  # noqa
        if capacity is not None:
            mask = _to_saliency_map(capacity.cpu().detach().numpy(),
                                    input_tensor.shape[1:],
                                    data_format="channels_first")
            input_mask_param = torch.tensor(mask).to(self.device)
            input_mask_param = input_mask_param.expand(input_tensor.shape[0],
                                                       -1, -1).unsqueeze(0)
        else:
            input_mask_param = torch.zeros(input_tensor.shape,
                                           dtype=torch.float).to(self.device)
        return nn.Parameter(input_mask_param, requires_grad=True)

    def init_mean_and_eps(self):
        mean = torch.tensor([0., 0., 0.]).view(1, -1, 1, 1).to(self.device)
        eps = torch.tensor([1., 1., 1.]).view(1, -1, 1, 1).to(self.device)
        return nn.Parameter(mean, requires_grad=True), nn.Parameter(
            eps, requires_grad=True)

    def forward(self, gaussian):
        noise = self.eps * gaussian + self.mean
        input_mask = F.sigmoid(self.input_mask_param)
        masked_input = input_mask * self.input_tensor + (1 - input_mask) * noise
        _ = self.context.classifier(masked_input)
        masked_feature_map = self.feature_map
        return masked_feature_map

    @torch.no_grad()
    def get_feature_map(self):
        _ = self.context.classifier(self.input_tensor.unsqueeze(0))
        return self.feature_map.squeeze(0)

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
                "Cannot detach hock. Either you never attached or already detached."
            )
