import torch

class Evaluation(object):
    """
    Base class for all evaluation methods
    get attribution map and image as input, returns a dictionary contains evaluation result
    """
    def eval(self, hmap: torch.Tensor, image: torch.Tensor) -> dict:
        raise NotImplementedError