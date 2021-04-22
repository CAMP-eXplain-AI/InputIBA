from abc import ABCMeta, abstractmethod
import torch.nn as nn
from contextlib import contextmanager
from ..utils import _InterruptExecution
import numpy as np


class BaseIBA(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 sigma=1.0,
                 initial_alpha=5.0,
                 input_mean=None,
                 input_std=None,
                 progbar=False,
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

        self.loss = []
        self.alpha_grads = []
        self.model_loss = []
        self.information_loss = []

        self.progbar = progbar
        self.reverse_lambda = reverse_lambda
        self.combine_loss = combine_loss
        self.device = device

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
                *args,
                **kwargs):
        pass

    def capacity(self):
        return self.buffer_capacity.mean(dim=0)

    def _get_saliency(self, mode='saliency'):
        assert mode in ('saliency', 'capacity'), f"mode should be either 'saliency' or " \
                                                 f"'capacity', but got {mode}"
        capacity_np = self.capacity().detach().cpu().numpy()
        if mode == 'saliency':
            return capacity_np.sum(1)
        else:
            return capacity_np / float(np.log(2))

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
        return -0.5 * (1 + log_var - mu ** 2 - log_var.exp())

    @abstractmethod
    @staticmethod
    def kl_div(*args, **kwargs):
        pass