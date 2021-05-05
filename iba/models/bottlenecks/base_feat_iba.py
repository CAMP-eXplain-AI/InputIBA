from abc import ABCMeta, abstractmethod
import torch
from .base_iba import BaseIBA
from ..utils import _InterruptExecution, _IBAForwardHook
from ..model_zoo import get_module
from contextlib import contextmanager


class BaseFeatureIBA(BaseIBA, metaclass=ABCMeta):

    def __init__(self,
                 context=None,
                 layer=None,
                 active_neurons_threshold=0.01,
                 estimator=None,
                 input_or_output='output',
                 relu=False,
                 sigma=1.0,
                 initial_alpha=5.0,
                 input_mean=None,
                 input_std=None,
                 reverse_lambda=False,
                 combine_loss=False,
                 device='cuda:0'):
        super(BaseFeatureIBA, self).__init__(
            sigma=sigma,
            initial_alpha=initial_alpha,
            input_mean=input_mean,
            input_std=input_std,
            reverse_lambda=reverse_lambda,
            combine_loss=combine_loss,
            device=device)
        assert (layer is None) ^ (context is None)
        self.layer = layer
        self.context = context
        self._active_neurons_threshold = active_neurons_threshold
        self.relu = relu
        self.initial_alpha = initial_alpha

        if estimator is None:
            self.estimator = self.reset_estimator()
        else:
            self.estimator = estimator
        self._estimate = False
        self.active_neurons = None

        self._hook_handle = None
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

    def detach(self):
        """ Remove the bottleneck to restore the original model """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        else:
            raise ValueError("Cannot detach hock. Either you never attached "
                             "or already detached.")

    @abstractmethod
    def init_alpha_and_kernel(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        """
        pass

    def forward(self, x):
        """You don't need to call this method manually. The iba acts as a
        model layer, passing the information in `x` along to the next layer
        either as-is or by restricting the flow of information. We use it
        also to estimate the distribution of `x` passing through the layer.
        """
        if self._restrict_flow:
            return self.do_restrict_info(x, self.alpha)
        if self._estimate:
            self.estimator(x)
        if self._interrupt_execution:
            raise _InterruptExecution()
        return x

    @contextmanager
    def enable_estimation(self):
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False

    @abstractmethod
    def reset_estimator(self):
        pass

    @abstractmethod
    def estimate(self,
                 model,
                 dataloader,
                 n_samples=10000,
                 verbose=False,
                 reset=True):
        pass

    @staticmethod
    def kl_div(r, lambda_, mean_r, std_r):
        r_norm = (r - mean_r) / std_r
        var_z = (1 - lambda_)**2
        log_var_z = torch.log(var_z)
        mu_z = r_norm * lambda_
        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity
