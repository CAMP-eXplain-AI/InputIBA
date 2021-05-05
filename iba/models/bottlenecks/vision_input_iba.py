import torch
import torch.nn as nn
from .base_input_iba import BaseInputIBA
from ..utils import _SpatialGaussianKernel, ifnone


class VisionInputIBA(BaseInputIBA):

    def __init__(self,
                 input_tensor,
                 input_mask,
                 sigma=1.0,
                 initial_alpha=5.0,
                 input_mean=None,
                 input_std=None,
                 reverse_lambda=False,
                 combine_loss=False,
                 device='cuda:0'):
        super(VisionInputIBA, self).__init__(input_tensor=input_tensor,
                                             input_mask=input_mask,
                                             sigma=sigma,
                                             initial_alpha=initial_alpha,
                                             input_mean=input_mean,
                                             input_std=input_std,
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

    def do_restrict_info(self, x, alpha):
        """ Selectively remove information from x by applying noise """
        if alpha is None:
            raise RuntimeWarning(
                "Alpha not initialized. Run _init() before using the bottleneck."
            )

        # Smoothen and expand alpha on batch dimension
        lamb = torch.sigmoid(alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # calculate kl divergence
        self.input_mean = ifnone(self.input_mean,
                                 torch.tensor(0.).to(self.device))
        self.input_std = ifnone(self.input_std,
                                torch.tensor(1.).to(self.device))
        self.buffer_capacity = self.kl_div(x, self.input_mask, lamb,
                                           self.input_mean, self.input_std)

        # apply mask on sampled x
        eps = x.data.new(x.size()).normal_()
        ε = self.input_std * eps + self.input_mean
        λ = lamb

        # TODO reverse_lambda and combined loss are only supported in original IBA
        # but might be also possible to implement here
        if self.reverse_lambda:
            # TODO rewrite
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive = λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε

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
            logger=None,
            log_every_steps=-1):
        assert input_tensor.shape[
            0] == 1, f"We can only fit one sample a time, but got {input_tensor.shape[0]}"
        batch = input_tensor.expand(batch_size, -1, -1, -1)

        # Reset from previous run or modifications
        self.reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        self.reset_loss_buffers()

        with self.restrict_flow():
            for i in range(opt_steps):
                optimizer.zero_grad()
                masked_img = self.forward(batch)
                cls_loss = model_loss_fn(masked_img)
                # Taking the mean is equivalent of scaling the sum with 1/K
                info_loss = self.capacity().mean()
                if self.reverse_lambda:
                    loss = -cls_loss + beta * info_loss
                else:
                    loss = cls_loss + beta * info_loss
                loss.backward(retain_graph=True)
                optimizer.step()

                self.loss_buffer.append(loss.item())
                self.cls_loss_buffer.append(cls_loss.item())
                self.info_loss_buffer.append(info_loss.item())
                if log_every_steps > 0 and (i + 1) % log_every_steps == 0:
                    log_str = f'Input IBA: step [{i + 1}/ {opt_steps}], '
                    log_str += f'loss: {self.loss_buffer[-1]:.5f}, '
                    log_str += f'cls loss: {self.cls_loss_buffer[-1]:.5f}, '
                    log_str += f'info loss: {self.info_loss_buffer[-1]:.5f}'
                    logger.info(log_str)

        return self._get_saliency(mode=mode)

    def detach(self):
        raise NotImplementedError
