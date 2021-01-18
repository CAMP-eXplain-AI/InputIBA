import numpy as np
import torch.nn as nn
import torch
import warnings
from contextlib import contextmanager
from torchvision.transforms import Normalize, Compose
from IBA.utils import _to_saliency_map, get_tqdm, ifnone

def to_saliency_map(capacity, shape=None):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape .

    Args:
        capacity (np.ndarray): Capacity in nats.
        shape (tuple): (height, width) of the image.
    """
    return _to_saliency_map(capacity, shape, data_format="channels_first")


class _SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels,):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, \
            "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              padding=0, kernel_size=kernel_size,
                              groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def parameters(self, **kwargs):
        """returns no parameters"""
        return []

    def forward(self, x):
        return self.conv(self.pad(x))


class Image_IBA(nn.Module):
    """
    Image IBA finds relevant features of your model by applying noise to
    image.

    Example: ::
        #TODO rewrite
        model = Net()
        # Create the Per-Sample Bottleneck:
        iba = IBA(model.conv4)

        img, target = next(iter(datagen(batch_size=1)))

        # Closure that returns the loss for one batch
        model_loss_closure = lambda x: F.nll_loss(F.log_softmax(model(x), target)

        # Explain class target for the given image
        saliency_map = iba.analyze(img.to(dev), model_loss_closure)
        plot_saliency_map(img.to(dev))


    Args:
        layer: The layer after which to inject the bottleneck
        sigma: The standard deviation of the gaussian kernel to smooth
            the mask, or None for no smoothing
        beta: Weighting of model loss and mean information loss.
        min_std: Minimum std of the features
        lr: Optimizer learning rate. default: 1. As we are optimizing
            over very few iterations, a relatively high learning rate
            can be used compared to the training of the model itself.
        batch_size: Number of samples to use per iteration
        input_or_output: Select either ``"output"`` or ``"input"``.
        initial_alpha: Initial value for the parameter.
    """
    def __init__(self,
                 image_mask=None,
                 image=None,
                 sigma=1.,
                 beta=10,
                 min_std=0.01,
                 img_eps_std=None,
                 img_eps_mean=None,
                 optimization_steps=10,
                 lr=1,
                 batch_size=10,
                 initial_alpha=5.0,
                 feature_mean=None,
                 feature_std=None,
                 progbar=False,
                 reverse_lambda=False,
                 combine_loss=False):
        super().__init__()
        self.beta = beta
        self.min_std = min_std
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.alpha = None  # Initialized on first forward pass
        self.image_mask = image_mask
        self.image = image
        self.img_eps_std = img_eps_std
        self.img_eps_mean = img_eps_mean
        self.progbar = progbar
        self.sigmoid = nn.Sigmoid()
        self._buffer_capacity = None  # Filled on forward pass, used for loss
        self.sigma = sigma
        self.device = None
        self._mean = feature_mean
        self._std = feature_std
        self._restrict_flow = False
        self.reverse_lambda = reverse_lambda
        self.combine_loss = combine_loss
        # initialize alpha
        if self.alpha is None:
            self._build()

    def _reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_alpha)

    def _build(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        """
        shape = self.image_mask.shape
        self.alpha = nn.Parameter(torch.full(shape, self.initial_alpha, device=self.device),
                                  requires_grad=True)
        if self.sigma is not None and self.sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = _SpatialGaussianKernel(kernel_size, self.sigma, shape[1]).to(self.device)
        else:
            self.smooth = None

    def forward(self, x):
        """
        You don't need to call this method manually.

        The IBA acts as a model layer, passing the information in `x` along to the next layer
        either as-is or by restricting the flow of infomration.
        We use it also to estimate the distribution of `x` passing through the layer.
        """
        if self._restrict_flow:
            return self._do_restrict_information(x, self.alpha)
        return x

    @staticmethod
    def _calc_capacity(mu, log_var):
        """ Return the feature-wise KL-divergence of p(z|x) and q(z) """
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    @staticmethod
    def _kl_div(x, g, image_mask, img_eps_mean, img_eps_std, lambda_, mean_x, std_x):
        """
        x:
        g:
        img_eps_mean:
        img_eps_std:
        image_mask: mask generated from GAN
        lambda_: learning parameter, image mask
        mean_x:
        std_x:

        """
        mean_x = 0
        std_x = 1
        r_norm = (x - mean_x + image_mask * (mean_x - g)) / ((1 - image_mask * lambda_) * std_x)
        var_z = (1 - lambda_)**2 / (1 - image_mask * lambda_)**2

        log_var_z = torch.log(var_z)

        mu_z = r_norm * lambda_

        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity

    def _do_restrict_information(self, g, alpha):
        """ Selectively remove information from x by applying noise """
        if alpha is None:
            raise RuntimeWarning("Alpha not initialized. Run _init() before using the bottleneck.")

        # Smoothen and expand alpha on batch dimension
        lamb = self.sigmoid(alpha)
        lamb = lamb.expand(g.shape[0], g.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # sample from random variable x
        eps = g.data.new(g.size()).normal_()
        ε_img = self.img_eps_std * eps + self.img_eps_mean
        # x = self.image_mask * g + (1-self.image_mask) * eps 
        x = g
        self.x = x

        # calculate kl divergence
        self._mean = ifnone(self._mean, torch.tensor(0.).to(self.device))
        self._std = ifnone(self._std, torch.tensor(1.).to(self.device))
        self._buffer_capacity = self._kl_div(x, g, self.image_mask, self.img_eps_mean, self.img_eps_std, lamb, self._mean, self._std)

        # apply mask on sampled x
        eps = x.data.new(x.size()).normal_()
        ε = self._std * eps + self._mean
        λ = lamb
        if self.reverse_lambda:
            #TODO rewrite
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive =  λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε

        return z


    @contextmanager
    def restrict_flow(self):
        """
        Context mananger to enable information supression.

        Example:
            To make a prediction, with the information flow being supressed.::

                with iba.restrict_flow():
                    # now noise is added
                    model(x)
        """
        self._restrict_flow = True
        try:
            yield
        finally:
            self._restrict_flow = False

    def analyze(self, input_t, model_loss_fn, mode="saliency",
                beta=None, optimization_steps=None, min_std=None,
                lr=None, batch_size=None):
        """
        Generates a heatmap for a given sample. Make sure you estimated mean and variance of the
        input distribution.

        Args:
            input_t: input image of shape (1, C, H W)
            model_loss_fn: closure evaluating the model
            mode: how to post-process the resulting map: 'saliency' (default) or 'capacity'
            beta: if not None, overrides the bottleneck beta value
            optimization_steps: if not None, overrides the bottleneck optimization_steps value
            min_std: if not None, overrides the bottleneck min_std value
            lr: if not None, overrides the bottleneck lr value
            batch_size: if not None, overrides the bottleneck batch_size value

        Returns:
            The heatmap of the same shape as the ``input_t``.
        """
        assert input_t.shape[0] == 1, "We can only fit one sample a time"

        # TODO: is None
        beta = ifnone(beta, self.beta)
        optimization_steps = ifnone(optimization_steps, self.optimization_steps)
        min_std = ifnone(min_std, self.min_std)
        lr = ifnone(lr, self.lr)
        batch_size = ifnone(batch_size, self.batch_size)

        batch = input_t.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self._reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        ## TODO no need to load mean and var
        # self._mean = self.img_eps_mean.clone()
        # std = self.img_eps_std.clone()
        # self._std = torch.max(std, min_std*torch.ones_like(std))

        self._loss = []
        self._alpha_grads = []
        self._model_loss = []
        self._information_loss = []

        opt_range = range(optimization_steps)
        try:
            tqdm = get_tqdm()
            opt_range = tqdm(opt_range, desc="Training Bottleneck", disable=not self.progbar)
        except ImportError:
            if self.progbar:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
                self.progbar = False

        with self.restrict_flow():
            for _ in opt_range:
                optimizer.zero_grad()
                masked_img = self.forward(batch)
                model_loss = model_loss_fn(masked_img)
                # Taking the mean is equivalent of scaling the sum with 1/K
                information_loss = self.capacity().mean()
                if self.reverse_lambda:
                    loss = -model_loss + beta * information_loss
                else:
                    loss = model_loss + beta * information_loss
                loss.backward(retain_graph=True)
                optimizer.step()

                self._alpha_grads.append(self.alpha.grad.cpu().numpy())
                self._loss.append(loss.item())
                self._model_loss.append(model_loss.item())
                self._information_loss.append(information_loss.item())

        return self._get_saliency(mode=mode, shape=input_t.shape[2:])

    def capacity(self):
        """
        Returns a tensor with the capacity from the last input, averaged
        over the redundant batch dimension.
        Shape is ``(self.channels, self.height, self.width)``
        """
        return self._buffer_capacity.mean(dim=0)

    def _get_saliency(self, mode='saliency', shape=None):
        capacity_np = self.capacity().detach().cpu().numpy()
        if mode == "saliency":
            # In bits, summed over channels, scaled to input
            return to_saliency_map(capacity_np, shape)
        elif mode == "capacity":
            # In bits, not summed, not scaled
            return capacity_np / float(np.log(2))
        else:
            raise ValueError