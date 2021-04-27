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
from skimage.transform import resize
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize


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


def tensor_to_np_img(img_t):
    """
    Convert a torch tensor of shape ``(c, h, w)`` to a numpy array of shape ``(h, w, c)``
    and reverse the torchvision prepocessing.
    """
    return Compose([
        Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        Normalize(std=[1, 1, 1], mean=[-0.485, -0.456, -0.406]),
    ])(img_t).detach().cpu().numpy().transpose(1, 2, 0)


class _SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """

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
            "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
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
        self.conv = nn.Conv2d(in_channels=channels,
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
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape .

    Args:
        capacity (np.ndarray): Capacity in nats.
        shape (tuple): (height, width) of the img.
    """
    return _to_saliency_map(capacity, shape, data_format="channels_first")


def _to_saliency_map(capacity, shape=None, data_format='channels_last'):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape.
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


def get_tqdm():
    """Tries to import ``tqdm`` from ``tqdm.auto`` if fails uses cli ``tqdm``."""
    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        from tqdm import tqdm
        return tqdm


def ifnone(a, b):
    """If a is None return b."""
    if a is None:
        return b
    else:
        return a


def to_unit_interval(x):
    """Scales ``x`` to be in ``[0, 1]``."""
    return (x - x.min()) / (x.max() - x.min())


def plot_saliency_map(saliency_map,
                      img=None,
                      ax=None,
                      colorbar_label='Bits / Pixel',
                      colorbar_fontsize=14,
                      min_alpha=0.2,
                      max_alpha=0.7,
                      vmax=None,
                      colorbar_size=0.3,
                      colorbar_pad=0.08):
    """
    Plots the heatmap with an bits/pixel colorbar and optionally overlays the img.

    Args:
        saliency_map (np.ndarray): the saliency_map.
        img (np.ndarray):  show this img under the saliency_map.
        ax: matplotlib axis. If ``None``, a new plot is created.
        colorbar_label (str): label for the colorbar.
        colorbar_fontsize (int): fontsize of the colorbar label.
        min_alpha (float): minimum alpha value for the overlay. only used if ``img`` is given.
        max_alpha (float): maximum alpha value for the overlay. only used if ``img`` is given.
        vmax: maximum value for colorbar.
        colorbar_size: width of the colorbar. default: Fixed(0.3).
        colorbar_pad: width of the colorbar. default: Fixed(0.08).

    Returns:
        The matplotlib axis ``ax``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed
    from skimage.color import rgb2grey, grey2rgb
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    if img is not None:
        # Underlay the img as greyscale
        grey = grey2rgb(rgb2grey(img))
        ax.imshow(grey)

    ax1_divider = make_axes_locatable(ax)
    if type(colorbar_size) == float:
        colorbar_size = Fixed(colorbar_size)
    if type(colorbar_pad) == float:
        colorbar_pad = Fixed(colorbar_pad)
    cax1 = ax1_divider.append_axes("right",
                                   size=colorbar_size,
                                   pad=colorbar_pad)
    if vmax is None:
        vmax = saliency_map.max()
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    n = 256
    half_jet_rgba = plt.cm.seismic(np.linspace(0.5, 1, n))
    half_jet_rgba[:, -1] = np.linspace(0.2, 1, n)
    cmap = mpl.colors.ListedColormap(half_jet_rgba)
    hmap_jet = cmap(norm(saliency_map))
    if img is not None:
        hmap_jet[:, :,
                 -1] = (max_alpha - min_alpha) * norm(saliency_map) + min_alpha
    ax.imshow(hmap_jet, alpha=max_alpha)
    cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm)
    cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')
    ax.set_frame_on(False)
    return ax
