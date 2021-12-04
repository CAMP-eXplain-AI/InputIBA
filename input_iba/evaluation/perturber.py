# code from IBA paper code
import numpy as np
import torch


class GridView:
    """ access something by 2D-tile indices """

    def __init__(self, orig_dim: tuple, tile_dim: tuple):
        self.orig_r = orig_dim[0]
        self.orig_c = orig_dim[1]
        self.tile_h = tile_dim[0]
        self.tile_w = tile_dim[1]
        self.tiles_r = self.orig_r // self.tile_h
        self.tiles_c = self.orig_c // self.tile_w
        self.grid = (self.tiles_r, self.tiles_c)

        if self.orig_r % self.tile_h != 0 or self.orig_c % self.tile_w != 0:
            print("Warning: GridView is not sound")

    def tile_slice(self, tile_r: int, tile_c: int):
        """ get the slice that would return the tile r,c """
        assert tile_r < self.tiles_r, \
            "tile {} is out of range with max {}".format(tile_r, self.tiles_r)
        assert tile_c < self.tiles_c, \
            "tile {} is out of range with max {}".format(tile_c, self.tiles_c)

        r = tile_r * self.tile_h
        c = tile_c * self.tile_w

        # get pixel indices of tile
        if tile_r == self.tiles_r - 1:
            slice_r = slice(r, None)
        else:
            slice_r = slice(r, r + self.tile_h)

        if tile_c == self.tiles_c - 1:
            slice_c = slice(c, None)
        else:
            slice_c = slice(c, c + self.tile_w)

        return slice_r, slice_c


class Perturber:

    def perturb(self, r: int, c: int):
        """ perturb a tile or pixel """
        raise NotImplementedError

    def get_current(self) -> np.ndarray:
        """ get current img with some perturbations """
        raise NotImplementedError

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        # TODO: might not needed, we determine perturb priority outside
        #  perturber
        """return a sorted list with shape length NUM_CELLS of
        which pixel/cell index to blur first"""
        raise NotImplementedError

    def get_grid_shape(self) -> tuple:
        """ return the shape of the grid, i.e. the max r, c values """
        raise NotImplementedError


class PixelPerturber(Perturber):

    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, r: int, c: int):
        self.current[:, r, c] = self.baseline[:, r, c]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.current.shape


class GridPerturber(Perturber):

    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor, tile_dim):
        assert len(tile_dim) == 2
        self.view = GridView(tuple(inp.shape[-2:]), tile_dim)
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, r: int, c: int):
        slc = self.view.tile_slice(r, c)
        self.current[:, slc[0], slc[1]] = self.baseline[:, slc[0], slc[1]]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.view.tiles_r, self.view.tiles_c

    def get_tile_shape(self) -> tuple:
        return self.view.tile_h, self.view.tile_w
