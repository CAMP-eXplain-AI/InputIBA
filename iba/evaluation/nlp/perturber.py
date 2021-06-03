import torch
import numpy as np


class Perturber:

    def perturb(self, r: int, c: int):
        """ perturb a tile or pixel """
        raise NotImplementedError

    def get_current(self) -> np.ndarray:
        """ get current img with some perturbations """
        raise NotImplementedError

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        # TODO: might not needed, we determine perturb priority outside perturber
        """return a sorted list with shape length NUM_CELLS of
        which pixel/cell index to blur first"""
        raise NotImplementedError

    def get_grid_shape(self) -> tuple:
        """ return the shape of the grid, i.e. the max r, c values """
        raise NotImplementedError


class WordPerturber():

    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, ind: int):
        self.current[ind, :] = self.baseline[ind, :]

    def get_current(self) -> torch.Tensor:
        return self.current.clone()