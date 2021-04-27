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
# THE SOFTWARE.                  F

import numpy as np
import torch.nn as nn
import torch
import warnings
from contextlib import contextmanager
from iba.models.utils import to_saliency_map, get_tqdm, ifnone, \
    _SpatialGaussianKernel, _InterruptExecution
from iba.models.estimators.estimators import VisionWelfordEstimator
from iba.models.model_zoo import get_module


class _IBAForwardHook:

    def __init__(self, iba, input_or_output="output"):
        self.iba = iba
        self.input_or_output = input_or_output

    def __call__(self, m, inputs, outputs):
        if self.input_or_output == "input":
            return self.iba(inputs)
        elif self.input_or_output == "output":
            return self.iba(outputs)


class IBA(nn.Module):
    """
    iba finds relevant features of your model by applying noise to
    intermediate features.

    Example: ::

        model = Net()
        # Create the Per-Sample Bottleneck:
        iba = iba(model.conv4)

        # Estimate the mean and variance.
        iba.estimate(model, datagen)

        img, target = next(iter(datagen(batch_size=1)))

        # Closure that returns the loss for one batch
        model_loss_closure = lambda x: F.nll_loss(F.log_softmax(model(x), target)

        # Explain class target for the given img
        saliency_map = iba.analyze(img.to(dev), model_loss_closure)
        plot_saliency_map(img.to(dev))


    Args:
        context (Attributor): the Attributor object which has a classifer(nn.Module) and a layer(
            str). If context is not None, register a forward hook for the context's classifier.
        layer (nn.Module, optional): The layer after which to inject the bottleneck. If layer is
            not None, register a forward hook for the layer.
        sigma: The standard deviation of the gaussian kernel to smooth
            the mask, or None for no smoothing
        input_or_output: Select either ``"output"`` or ``"input"``.
        initial_alpha: Initial value for the parameter.
    """

    def __init__(self,
                 layer=None,
                 context=None,
                 sigma=1.,
                 active_neurons_threshold=0.01,
                 initial_alpha=5.0,
                 feature_mean=None,
                 feature_std=None,
                 estimator=None,
                 progbar=False,
                 input_or_output="output",
                 relu=False,
                 reverse_lambda=False,
                 combine_loss=False):
        super().__init__()
        assert (layer is None) ^ (context is None)
        self.layer = layer
        self.context = context
        self._active_neurons_threshold = active_neurons_threshold
        self.relu = relu
        self.initial_alpha = initial_alpha
        self.alpha = None  # Initialized on first forward pass
        self.progbar = progbar
        self.sigmoid = nn.Sigmoid()
        self._buffer_capacity = None  # Filled on forward pass, used for loss
        self.sigma = sigma
        self.estimator = ifnone(estimator, VisionWelfordEstimator())
        self.device = None
        self._estimate = False
        self._mean = feature_mean
        self._std = feature_std
        self._active_neurons = None
        self._restrict_flow = False
        self._interrupt_execution = False
        self._hook_handle = None
        self.reverse_lambda = reverse_lambda
        self.combine_loss = combine_loss

        # Attach the bottleneck after the model layer as forward hook
        if self.context is not None:
            self._hook_handle = get_module(
                self.context.classifier,
                self.context.layer).register_forward_hook(
                    _IBAForwardHook(self, input_or_output))
        elif self.layer is not None:
            self._hook_handle = self.layer.register_forward_hook(
                _IBAForwardHook(self, input_or_output))
        else:
            raise ValueError(
                'context and layer cannot be None at the same time')

    def reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_alpha)

    def init_alpha_and_kernel(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        """
        if self.estimator.n_samples() <= 0:
            raise RuntimeWarning("You need to estimate the feature distribution"
                                 " before using the bottleneck.")
        shape = self.estimator.shape
        device = self.estimator.device
        self.alpha = nn.Parameter(torch.full(shape,
                                             self.initial_alpha,
                                             device=device),
                                  requires_grad=True)
        if self.sigma is not None and self.sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(
                2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = _SpatialGaussianKernel(kernel_size, self.sigma,
                                                 shape[0]).to(device)
        else:
            self.smooth = None

    def detach(self):
        """ Remove the bottleneck to restore the original model """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        else:
            raise ValueError(
                "Cannot detach hock. Either you never attached or already detached."
            )

    def forward(self, x):
        """
        You don't need to call this method manually.

        The iba acts as a model layer, passing the information in `x` along to the next layer
        either as-is or by restricting the flow of infomration.
        We use it also to estimate the distribution of `x` passing through the layer.
        """
        if self._restrict_flow:
            return self._do_restrict_information(x, self.alpha)
        if self._estimate:
            self.estimator(x)
        if self._interrupt_execution:
            raise _InterruptExecution()
        return x

    @contextmanager
    def interrupt_execution(self):
        """
        Interrupts the execution of the model, once PerSampleBottleneck is called. Useful
        for estimation when the model has only be executed until the Per-Sample Bottleneck.

        Example:
            Executes the model only until the bottleneck layer::

                with bltn.interrupt_execution():
                    out = model(x)
                    # out will not be defined
                    print("this will not be printed")
        """
        self._interrupt_execution = True
        try:
            yield
        except _InterruptExecution:
            pass
        finally:
            self._interrupt_execution = False

    @staticmethod
    def _calc_capacity(mu, log_var):
        """ Return the feature-wise KL-divergence of p(z|x) and q(z) """
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    @staticmethod
    def _kl_div(r, lambda_, mean_r, std_r):
        r_norm = (r - mean_r) / std_r
        var_z = (1 - lambda_)**2

        log_var_z = torch.log(var_z)

        mu_z = r_norm * lambda_

        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity

    def _do_restrict_information(self, x, alpha):
        """ Selectively remove information from x by applying noise """
        if alpha is None:
            raise RuntimeWarning(
                "Alpha not initialized. Run _init() before using the bottleneck."
            )

        if self._mean is None:
            self._mean = self.estimator.mean()

        if self._std is None:
            self._std = self.estimator.std()

        if self._active_neurons is None:
            self._active_neurons = self.estimator.active_neurons()

        # Smoothen and expand alpha on batch dimension
        lamb = self.sigmoid(alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        self._buffer_capacity = self._kl_div(x, lamb, self._mean,
                                             self._std) * self._active_neurons

        eps = x.data.new(x.size()).normal_()
        ε = self._std * eps + self._mean
        λ = lamb
        if self.reverse_lambda:
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive = λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε
        z *= self._active_neurons

        # Sample new output values from p(z|x)

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z

    @contextmanager
    def enable_estimation(self):
        """
        Context manager to enable estimation of the mean and standard derivation.
        We recommend to use the `self.estimate` method.
        """
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False

    def reset_estimate(self):
        """
        Resets the estimator. Useful if the distribution changes. Which can happen if you
        trained the model more.
        """
        self.estimator = VisionWelfordEstimator()

    def estimate(self,
                 model,
                 dataloader,
                 device=None,
                 n_samples=10000,
                 progbar=None,
                 reset=True):
        """ Estimate mean and variance using the welford estimator.
            Usually, using 10.000 i.i.d. samples gives decent estimates.

            Args:
                model: the model containing the bottleneck layer
                dataloader: yielding ``batch``'s where the first sample
                    ``batch[0]`` is the img batch.
                device: images will be transfered to the device. If ``None``, it uses the device
                    of the first model parameter.
                n_samples (int): run the estimate on that many samples
                progbar (bool): show a progress bar.
                reset (bool): reset the current estimate of the mean and std

        """
        progbar = progbar if progbar is not None else self.progbar
        if progbar:
            try:
                tqdm = get_tqdm()
                bar = tqdm(dataloader, total=n_samples)
            except ImportError:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
                bar = None
        else:
            bar = None

        if device is None:
            device = next(iter(model.parameters())).device
        if reset:
            self.reset_estimate()
        for batch in dataloader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                imgs = batch[0]
            else:
                imgs = batch['img']
            if self.estimator.n_samples() > n_samples:
                break
            with torch.no_grad(), self.interrupt_execution(
            ), self.enable_estimation():
                model(imgs.to(device))
            if bar:
                bar.update(len(imgs))
        if bar:
            bar.close()

        # Cache results
        self._mean = self.estimator.mean()
        self._std = self.estimator.std()
        self._active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        # After estimaton, feature map dimensions are known and
        # we can initialize alpha and the smoothing kernel
        if self.alpha is None:
            self.init_alpha_and_kernel()

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

    def analyze(self,
                input_t,
                model_loss_fn,
                mode="saliency",
                beta=10.0,
                opt_steps=10,
                min_std=0.01,
                lr=1.0,
                batch_size=10):
        """
        Generates a heatmap for a given sample. Make sure you estimated mean and variance of the
        input distribution.

        Args:
            input_t: input img of shape (1, C, H W)
            model_loss_fn: closure evaluating the model
            mode: how to post-process the resulting map: 'saliency' (default) or 'capacity'
            beta: beta of the combined loss.
            opt_steps: optimization steps.
            min_std: minimal standard deviation.
            lr: learning rate.
            batch_size: batch size.
        Returns:
            The heatmap of the same shape as the ``input_t``.
        """
        assert input_t.shape[0] == 1, "We can only fit one sample a time"

        batch = input_t.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self.reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.estimator.n_samples() < 1000:
            warnings.warn(
                f"Selected estimator was only fitted on {self.estimator.n_samples()} "
                f"samples. Might not be enough! We recommend 10.000 samples.")
        std = self.estimator.std()
        self._active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        self._std = torch.max(std, min_std * torch.ones_like(std))

        self._loss = []
        self._alpha_grads = []
        self._model_loss = []
        self._information_loss = []

        opt_range = range(opt_steps)
        try:
            tqdm = get_tqdm()
            opt_range = tqdm(opt_range,
                             desc="Training Bottleneck",
                             disable=not self.progbar)
        except ImportError:
            if self.progbar:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
                self.progbar = False

        with self.restrict_flow():
            for _ in opt_range:
                optimizer.zero_grad()
                model_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling the sum with 1/K
                information_loss = self.capacity().mean()
                if self.reverse_lambda:
                    loss = -model_loss + beta * information_loss
                else:
                    loss = model_loss + beta * information_loss
                loss.backward()
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
