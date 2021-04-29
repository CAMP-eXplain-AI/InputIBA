import torch
import torch.nn as nn
import warnings
from ..utils import get_tqdm, _SpatialGaussianKernel
from .base_feat_iba import BaseFeatureIBA
from ..estimators import VisionWelfordEstimator


class VisionFeatureIBA(BaseFeatureIBA):
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
    """

    def __init__(self, **kwargs):
        super(VisionFeatureIBA, self).__init__(**kwargs)

    @torch.no_grad()
    def reset_alpha(self):
        self.alpha.fill_(self.initial_alpha)

    def init_alpha_and_kernel(self):
        # TODO to check if it is neccessary to keep it in base class
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

    def reset_estimator(self):
        self.estimator = VisionWelfordEstimator()

    def estimate(self,
                 model,
                 dataloader,
                 n_samples=10000,
                 progbar=False,
                 reset=True):
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

        if reset:
            self.reset_estimator()
        for batch in dataloader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                imgs = batch[0]
            else:
                imgs = batch['input']
            if self.estimator.n_samples() > n_samples:
                break
            with torch.no_grad(), self.interrupt_execution(
            ), self.enable_estimation():
                model(imgs.to(self.device))
            if bar:
                bar.update(len(imgs))
        if bar:
            bar.close()

        # Cache results
        self.input_mean = self.estimator.mean()
        self.input_std = self.estimator.std()
        self.active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        # After estimaton, feature map dimensions are known and
        # we can initialize alpha and the smoothing kernel
        if self.alpha is None:
            self.init_alpha_and_kernel()

    def do_restrict_info(self, x, alpha):
        if alpha is None:
            raise RuntimeWarning(
                "Alpha not initialized. Run _init() before using the bottleneck."
            )

        if self.input_mean is None:
            self.input_mean = self.estimator.mean()

        if self.input_std is None:
            self.input_std = self.estimator.std()

        if self.active_neurons is None:
            self.active_neurons = self.estimator.active_neurons()

        # Smoothen and expand alpha on batch dimension
        lamb = torch.sigmoid(alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        self.buffer_capacity = self.kl_div(x, lamb, self.input_mean,
                                           self.input_std) * self.active_neurons

        eps = x.data.new(x.size()).normal_()
        ε = self.input_std * eps + self.input_mean
        λ = lamb
        if self.reverse_lambda:
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive = λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε
        z *= self.active_neurons

        # Sample new output values from p(z|x)

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z

    def analyze(  # noqa
            self,
            input_tensor,
            model_loss_fn,
            mode='saliency',
            beta=10.0,
            opt_steps=10,
            lr=1.0,
            batch_size=10,
            min_std=0.01):
        assert input_tensor.shape[0] == 1, "We can only fit one sample a time"

        batch = input_tensor.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self.reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.estimator.n_samples() < 1000:
            warnings.warn(
                f"Selected estimator was only fitted on {self.estimator.n_samples()} "
                f"samples. Might not be enough! We recommend 10.000 samples.")
        std = self.estimator.std()
        self.active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        self.input_std = torch.max(std, min_std * torch.ones_like(std))

        self.loss = []
        self.alpha_grads = []
        self.model_loss = []
        self.information_loss = []

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

                self.alpha_grads.append(self.alpha.grad.cpu().numpy())
                self.loss.append(loss.item())
                self.model_loss.append(model_loss.item())
                self.information_loss.append(information_loss.item())

        return self._get_saliency(mode=mode, shape=input_tensor.shape[2:])
