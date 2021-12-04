from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn
from contextlib import contextmanager

from ..utils import _InterruptExecution, to_saliency_map


class BaseIBA(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 sigma=1.0,
                 initial_alpha=5.0,
                 input_mean=None,
                 input_std=None,
                 reverse_lambda=False,
                 combine_loss=False,
                 device='cuda:0'):
        super(BaseIBA, self).__init__()
        self._restrict_flow = False
        self._interrupt_execution = False

        self.buffer_capacity = None
        self.sigma = sigma
        self.initial_alpha = initial_alpha
        self.input_mean = input_mean
        self.input_std = input_std
        # alpha is initialized on the first forward pass
        self.alpha = None
        self.smooth = None

        self.loss_buffer = []
        self.cls_loss_buffer = []
        self.info_loss_buffer = []

        self.reverse_lambda = reverse_lambda
        self.combine_loss = combine_loss
        self.device = device

    def reset_loss_buffers(self):
        self.loss_buffer.clear()
        self.cls_loss_buffer.clear()
        self.info_loss_buffer.clear()

    def get_loss_history(self):
        return self.loss_buffer

    def get_cls_loss_history(self):
        return self.cls_loss_buffer

    def get_info_loss_history(self):
        return self.info_loss_buffer

    @abstractmethod
    def reset_alpha(self):
        pass

    @abstractmethod
    def init_alpha_and_kernel(self):
        pass

    @abstractmethod
    def detach(self):
        pass

    @abstractmethod
    def do_restrict_info(self, x, alpha):
        pass

    @abstractmethod
    def analyze(self,
                input_tensor,
                model_loss_fn,
                mode='saliency',
                beta=10.0,
                opt_steps=10,
                lr=1.0,
                batch_size=10,
                logger=None,
                log_every_steps=-1,
                *args,
                **kwargs):
        pass

    def capacity(self):
        return self.buffer_capacity.mean(dim=0)

    def _get_saliency(self, mode='saliency', shape=None):
        assert mode in ('saliency', 'capacity'), \
            f"mode should be either 'saliency' or capacity', but got {mode}"
        capacity_np = self.capacity().detach().cpu().numpy()
        if mode == 'saliency':
            return to_saliency_map(capacity_np, shape)
        else:
            return capacity_np / float(np.log(2))

    @contextmanager
    def interrupt_execution(self):
        """
        Interrupts the execution of the model, once PerSampleBottleneck is
        called. Useful for estimation when the model has only be executed
        until the Per-Sample Bottleneck.

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

    @contextmanager
    def restrict_flow(self):
        self._restrict_flow = True
        try:
            yield
        finally:
            self._restrict_flow = False

    @staticmethod
    def calc_capacity(mu, log_var):
        """ Return the feature-wise KL-divergence of p(z|x) and q(z) """
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    @staticmethod
    @abstractmethod
    def kl_div(*args, **kwargs):
        pass
