import mmcv
import torch
import torch.nn as nn
import warnings
from ..utils import _SpatialGaussianKernel
from .base_feat_iba import BaseFeatureIBA
from ..estimators import NLPWelfordEstimator
from tqdm import tqdm


class NLPFeatureIBA(BaseFeatureIBA):
    """
        iba finds relevant features of your model by applying noise to
        intermediate features.
    """

    def __init__(self, **kwargs):
        super(NLPFeatureIBA, self).__init__(**kwargs)

    @torch.no_grad()
    def reset_alpha(self):
        self.alpha.fill_(self.initial_alpha)

    def init_alpha_and_kernel(self):
        # TODO to check if it is necessary to keep it in base class, currently found no difference
        if self.estimator.n_samples() <= 0:
            raise RuntimeWarning(
                "You need to estimate the feature distribution"
                " before using the bottleneck.")
        shape = self.estimator.shape
        device = self.estimator.device
        self.alpha = nn.Parameter(
            torch.full(shape, self.initial_alpha, device=device),
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
        self.estimator = NLPWelfordEstimator()

    def estimate(self,
                 model,
                 dataloader,
                 n_samples=10000,
                 verbose=False,
                 reset=True):
        if verbose:
            bar = tqdm(dataloader, total=n_samples)
        else:
            bar = None

        if reset:
            self.reset_estimator()
        for batch in dataloader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                (label, text, text_lengths, fname) = batch
            else:
                # if torchtext.legacy is used for dataset
                text, text_lengths = batch.text
            if self.estimator.n_samples() > n_samples:
                break
            with torch.no_grad(), self.interrupt_execution(
            ), self.enable_estimation():
                model(text.to(self.device), text_lengths)
            if bar:
                bar.update(len(text_lengths))
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

        # get output
        output, hidden_and_cell = x
        output_padded, text_lengths = nn.utils.rnn.pad_packed_sequence(output)

        # Smoothen and expand alpha on batch dimension
        lamb = self.sigmoid(alpha)
        lamb = lamb.expand(output_padded.shape[0], 1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        self._buffer_capacity = self._kl_div(output_padded, lamb, self._mean.expand(output_padded.shape[0], 1, -1),
                                             self._std.expand(output_padded.shape[0], 1, -1)) * self._active_neurons

        eps = output_padded.data.new(output_padded.size()).normal_()
        ε = self._std * eps + self._mean
        λ = lamb
        if self.reverse_lambda:
            z = λ * ε + (1 - λ) * output_padded
        elif self.combine_loss:
            z_positive = λ * output_padded + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * output_padded
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * output_padded + (1 - λ) * ε
        z *= self._active_neurons

        # Sample new output values from p(z|x)

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        # pack value again to pass to later layer
        z_packed = (nn.utils.rnn.pack_padded_sequence(z, text_lengths), hidden_and_cell)
        return z_packed

    def analyze(  # noqa
            self,
            input_tensor,
            model_loss_fn,
            mode='saliency',
            beta=10.0,
            opt_steps=10,
            lr=1.0,
            batch_size=10,
            min_std=0.01,
            logger=None,
            log_every_steps=-1):
        assert input_tensor.shape[1] == 1, \
            f"We can only fit one sample a time, " \
            f"but got {input_tensor.shape[1]} samples"
        if logger is None:
            logger = mmcv.get_logger('iba')

        batch = input_tensor.expand(-1, batch_size)

        # Reset from previous run or modifications
        self.reset_alpha(input_tensor.shape[0])
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.estimator.n_samples() < 1000:
            warnings.warn(f"Selected estimator was only fitted "
                          f"on {self.estimator.n_samples()} samples. Might "
                          f"not be enough! We recommend 10.000 samples.")
        std = self.estimator.std()
        self.active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        self.input_std = torch.max(std, min_std * torch.ones_like(std))

        self.reset_loss_buffers()

        with self.restrict_flow():
            for i in range(opt_steps):
                optimizer.zero_grad()
                cls_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling the sum with 1/K
                info_loss = self.capacity().mean()
                if self.reverse_lambda:
                    loss = -cls_loss + beta * info_loss
                else:
                    loss = cls_loss + beta * info_loss
                loss.backward()
                optimizer.step()

                self.loss_buffer.append(loss.item())
                self.cls_loss_buffer.append(cls_loss.item())
                self.info_loss_buffer.append(info_loss.item())
                if log_every_steps > 0 and (i + 1) % log_every_steps == 0:
                    log_str = f'Feature IBA: step [{i + 1}/ {opt_steps}], '
                    log_str += f'loss: {self.loss_buffer[-1]:.5f}, '
                    log_str += f'cls loss: {self.cls_loss_buffer[-1]:.5f}, '
                    log_str += f'info loss: {self.info_loss_buffer[-1]:.5f}'
                    logger.info(log_str)

        return self._get_saliency(mode=mode, shape=input_tensor.shape[2:])
