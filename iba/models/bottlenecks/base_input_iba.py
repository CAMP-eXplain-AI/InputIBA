from .base_iba import BaseIBA


class BaseInputIBA(BaseIBA):
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
        super(BaseInputIBA, self).__init__(sigma=sigma,
                                           initial_alpha=initial_alpha,
                                           input_mean=input_mean,
                                           input_std=input_std,
                                           progbar=progbar,
                                           reverse_lambda=reverse_lambda,
                                           combine_loss=combine_loss,
                                           device=device)
        self.input = input
        self.input_mask = input_mask
        self.input_eps_mean = input_eps_mean
        self.input_eps_std = input_eps_std
