# Copyright (c) Karl Schulz, Leon Sixt
#
# All rights reserved.
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize


class _IBAForwardHook:

    def __init__(self, iba, input_or_output="output"):
        self.iba = iba
        self.input_or_output = input_or_output

    def __call__(self, m, inputs, outputs):
        if self.input_or_output == "input":
            return self.iba(inputs)
        elif self.input_or_output == "output":
            return self.iba(outputs)


class _InterruptExecution(Exception):
    pass


class _SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels,
    used to smoothen the input
    """

    def __init__(
        self,
        kernel_size,
        sigma,
        channels,
    ):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, \
            "kernel_size must be an odd number (for padding), " \
            "{} given".format(self.kernel_size)
        variance = sigma**2.
        x_cord = torch.arange(kernel_size, dtype=torch.float)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * np.pi * variance)) * torch.exp(-torch.sum(
            (xy_grid - mean_xy)**2., dim=-1) / (2 * variance))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1,
                                     -1)  # expand in channel dimension
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            padding=0,
            kernel_size=kernel_size,
            groups=channels,
            bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def parameters(self, **kwargs):
        """returns no parameters"""
        return []

    def forward(self, x):
        return self.conv(self.pad(x))


def to_saliency_map(capacity, shape=None):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of
    the given shape .

    Args:
        capacity (np.ndarray): Capacity in nats.
        shape (tuple): (height, width) of the img.
    """
    return _to_saliency_map(capacity, shape, data_format="channels_first")


def _to_saliency_map(capacity, shape=None, data_format='channels_last'):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the
    given shape.
    PyTorch:    Use data_format == 'channels_first'
    Tensorflow: Use data_format == 'channels_last'
    """
    if data_format == 'channels_first':
        saliency_map = np.nansum(capacity, 0)
    elif data_format == 'channels_last':
        saliency_map = np.nansum(capacity, -1)
    else:
        raise ValueError

    # to bits
    saliency_map /= float(np.log(2))

    if shape is not None:
        ho, wo = saliency_map.shape
        h, w = shape
        # Scale bits to the pixels
        saliency_map *= (ho * wo) / (h * w)
        return resize(saliency_map, shape, order=1, preserve_range=True)
    else:
        return saliency_map


def ifnone(a, b):
    """If a is None return b."""
    if a is None:
        return b
    else:
        return a
