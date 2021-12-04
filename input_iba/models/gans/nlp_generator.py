import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

from ..model_zoo import get_module
from .base_generator import BaseGenerator
from .builder import GENERATORS


class _ForwardHookWrapper:

    def __init__(self, class_to_wrap, input_or_output="output"):
        self.class_to_wrap = class_to_wrap
        self.input_or_output = input_or_output

    def __call__(self, m, inputs, outputs):
        if self.input_or_output == "input":
            return self.class_to_wrap(inputs)
        elif self.input_or_output == "output":
            return self.class_to_wrap(outputs)


class WordEmbeddingMasker(nn.Module):

    def __init__(self, mean, eps, word_embedding_mask_param):
        super().__init__()
        self.eps = eps
        self.mean = mean
        self.word_embedding_mask_param = word_embedding_mask_param
        self.sigmoid = nn.Sigmoid()
        self.gaussian = None

    def set_gaussian_noise(self, gaussian):
        """set gaussian noise"""
        self.gaussian = gaussian

    def forward(self, x):
        if self.gaussian is None:
            return x
        noise = self.eps * self.gaussian + self.mean
        word_mask = self.sigmoid(self.word_embedding_mask_param)
        z = word_mask * x + (1 - word_mask) * noise
        return z


@GENERATORS.register_module()
class NLPGenerator(BaseGenerator):
    """
    Generator takes random noise as input, learnable parameter is the img mask.
    masked img (with noise added) go through the original network and generate
    masked feature map.
    """

    def __init__(self, input_tensor, context, device='cuda:0', capacity=None):
        super().__init__(input_tensor, context, device=device)
        self.input_tensor = input_tensor
        # TODO make img_mask_param a Parameter
        if capacity is not None:
            # TODO review
            word_embedding_mask_param = torch.tensor(
                capacity.sum(1).cpu().detach().numpy()).to(device)
            # TODO pass embedding dim from attributer
            word_embedding_mask_param = word_embedding_mask_param.unsqueeze(
                -1).unsqueeze(-1).expand(-1, 1, 100).clone()
            self.word_embedding_mask_param = word_embedding_mask_param
        else:
            self.word_embedding_mask_param = torch.zeros(
                input_tensor.shape, dtype=torch.float).to(device)
        self.word_embedding_mask_param.requires_grad = True
        # TODO make mean and eps Parameters.
        self.mean = torch.zeros(
            (self.word_embedding_mask_param.shape[0], 1, 1)).to(device)
        self.mean.requires_grad = True
        self.eps = torch.ones(
            (self.word_embedding_mask_param.shape[0], 1, 1)).to(device)
        self.eps.requires_grad = True
        self.feature_map = None

        # register hook in trained classification network to get hidden
        # representation of masked input
        def store_feature_map(model, input, output):
            self.feature_map = output

        self._hook_handle = get_module(
            self.context.classifier,
            self.context.layer).register_forward_hook(store_feature_map)

        # construct word embedding masker
        self.masker = WordEmbeddingMasker(self.mean, self.eps,
                                          self.word_embedding_mask_param)

        # register hook to mask word embedding
        layer = get_module(self.context.classifier, "embedding")
        self._mask_hook_handle = layer.register_forward_hook(
            _ForwardHookWrapper(self.masker, 'output'))

        # placeholder for input mask parameters
        self.input_mask_param = self.word_embedding_mask_param

    def forward(self, gaussian):
        self.masker.set_gaussian_noise(gaussian)
        input_tensor = self.input_tensor.unsqueeze(1).expand(
            -1, gaussian.shape[1])
        text_lengths = torch.tensor([self.input_tensor.shape[0]
                                     ]).expand(gaussian.shape[1]).to('cpu')

        _ = self.context.classifier(input_tensor, text_lengths)
        feature_map_padded, feature_map_lengths = pad_packed_sequence(
            self.feature_map[0])
        return feature_map_padded

    @torch.no_grad()
    def get_feature_map(self):
        text_length = torch.tensor([self.input_tensor.shape[0]
                                    ]).expand(1).to('cpu')
        _ = self.context.classifier(
            self.input_tensor.unsqueeze(1), text_length)
        feature_map_padded, feature_map_lengths = pad_packed_sequence(
            self.feature_map[0])
        return feature_map_padded

    def clear(self):
        del self.feature_map
        self.feature_map = None
        self.detach()

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._mask_hook_handle.remove()
            self._mask_hook_handle = None
        else:
            raise ValueError("Cannot detach hock. Either you never attached "
                             "or already detached.")
