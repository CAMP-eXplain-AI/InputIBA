import torch
from iba.evaluation.base import Evaluation


class SensitivityN(Evaluation):
    def __init__(self, n=None):
        self.n = n

    def eval(self, hmap: torch.Tensor, image: torch.Tensor) -> dict:
        pass
