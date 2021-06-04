import torch


class WordPerturber:
    # TODO fix the inheritance

    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, ind: int):
        self.current[ind, :] = self.baseline[ind, :]

    def get_current(self) -> torch.Tensor:
        return self.current.clone()