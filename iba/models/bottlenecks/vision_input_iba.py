import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_input_iba import BaseInputIBA
from ..utils import _SpatialGaussianKernel, to_saliency_map, get_tqdm, ifnone
import warnings


class VisionInputIBA(BaseInputIBA):
    def __init__(self,
                 input,
                 input_mask,
                 input_eps_mean=0.0,
                 input_eps_std=1.0,
                 sigma=1.0,
                 initial_alpha=5.0,
                 input_mean=None,
                 input_std=None,
                 progbar=False,
                 reverse_lambda=False,
                 combine_loss=False,
                 device='cuda:0'):
        super(VisionInputIBA, self).__init__(input=input,
                                             input_mask=input_mask,
                                             input_eps_mean=input_eps_mean,
                                             input_eps_std=input_eps_std,
                                             sigma=sigma,
                                             initial_alpha=initial_alpha,
                                             input_mean=input_mean,
                                             input_std=input_std,
                                             progbar=progbar,
                                             reverse_lambda=reverse_lambda,
                                             combine_loss=combine_loss,
                                             device=device)
        if self.alpha is None:
            self.init_alpha_and_kernel()

    @torch.no_grad()
    def reset_alpha(self):
        self.alpha.fill_(self.initial_alpha)

    def init_alpha_and_kernel(self):
        shape = self.input_mask.shape
        self.alpha = nn.Parameter(torch.full(shape,
                                             self.initial_alpha,
                                             device=self.device),
                                  requires_grad=True)
        if self.sigma is not None and self.sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(
                2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = _SpatialGaussianKernel(kernel_size, self.sigma,
                                                 shape[1]).to(self.device)
        else:
            self.smooth = None

    def forward(self, x):
        if self._restrict_flow:
            return self.do_restrict_info(x, self.alpha)
        return x

    def kl_div(self,
               x,
               g,
               input_mask,
               lambda_):
        # TODO
        mean_x = 0
        std_x = 1
        r_norm = (x - mean_x + input_mask *
                  mean_x) / ((1 - input_mask * lambda_) * std_x)
        var_z = (1 - lambda_) ** 2 / (1 - input_mask * lambda_) ** 2

        log_var_z = torch.log(var_z)

        mu_z = r_norm * lambda_

        capacity = -0.5 * (1 + log_var_z - mu_z ** 2 - var_z)
        return capacity

    def do_restrict_info(self, x, alpha):
        if alpha is None:
            raise RuntimeWarning(
                "Alpha not initialized. Run _init() before using the bottleneck."
            )

        # Smoothen and expand alpha on batch dimension
        lamb = F.sigmoid(alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # sample from random variable x
        eps = x.data.new(x.size()).normal_()
        ε_img = self.img_eps_std * eps + self.img_eps_mean

        # calculate kl divergence
        self.input_mean = ifnone(self._mean, torch.tensor(0.).to(self.device))
        self.input_std = ifnone(self._std, torch.tensor(1.).to(self.device))
        # TODO
        self.buffer_capacity = self.kl_div(x, self.input_mask, lamb)

        # apply mask on sampled x
        eps = x.data.new(x.size()).normal_()
        ε = self._std * eps + self._mean
        λ = lamb
        if self.reverse_lambda:
            #TODO rewrite
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive = λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε

        return z

    def analyze(self,
                input,
                model_loss_fn,
                mode='saliency',
                beta=10.0,
                opt_steps=10,
                lr=1.0,
                batch_size=10):
        assert input.shape[0] == 1, "We can only fit one sample a time"
        batch = input.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self.reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

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

                self.alpha_grads.append(self.alpha.grad.cpu().numpy())
                self.loss.append(loss.item())
                self.model_loss.append(model_loss.item())
                self.information_loss.append(information_loss.item())

        return self._get_saliency(mode=mode)