import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import trapezoid

from input_iba.evaluation.base import BaseEvaluation
from input_iba.evaluation.perturber import GridPerturber


class Degradation(BaseEvaluation):

    def __init__(self, model, target, tile_size, store_imgs=False, batch=8):
        self.model = model
        self.batch = batch
        self.target = target
        self.morf_scores = []
        self.lerf_scores = []
        self.tile_size = tile_size
        self.img_history_morf = [] if store_imgs else None
        self.img_history_lerf = [] if store_imgs else None

    @torch.no_grad()
    def evaluate(self, heatmap: torch.Tensor, image: torch.Tensor) -> dict:
        self.model.eval()

        # compress heatmap to 2D if needed
        if heatmap.ndim == 3:
            heatmap = heatmap.mean(0)

        # get 2d tile attribution
        perturber = GridPerturber(image, torch.zeros_like(image),
                                  self.tile_size)
        grid_heatmap = torch.zeros(perturber.get_grid_shape())
        for r in range(grid_heatmap.shape[0]):
            for c in range(grid_heatmap.shape[1]):
                grid_heatmap[r][c] = heatmap[perturber.view.tile_slice(
                    r, c)].sum()

        # sort tile in attribution
        num_pixels = torch.numel(grid_heatmap)
        _, indices = torch.topk(grid_heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), grid_heatmap.size())
        _, reverse_indices = torch.topk(
            grid_heatmap.flatten(), num_pixels, largest=False)
        reverse_indices = np.unravel_index(reverse_indices.cpu().numpy(),
                                           grid_heatmap.size())

        # TODO to make it compatible with multi-label classification setting
        # TODO to make baseline_score and morf_scores local variables
        #  rather than object attributes
        # get baseline score
        self.baseline_score = F.softmax(
            self.model(
                image.unsqueeze(0).to(next(
                    self.model.parameters()).device)))[:, self.target]
        self.baseline_score = self.baseline_score.detach().cpu().numpy()

        # apply deletion game using MoRF
        print("MoRF deletion")
        self.morf_scores = self._procedure_perturb(perturber, num_pixels,
                                                   indices,
                                                   self.img_history_morf)
        MoRF_img = perturber.get_current()

        # apply deletion game using LeRF
        perturber = GridPerturber(image, torch.zeros_like(image),
                                  self.tile_size)
        print("LeRF deletion")
        self.lerf_scores = self._procedure_perturb(perturber, num_pixels,
                                                   reverse_indices,
                                                   self.img_history_lerf)
        LeRF_img = perturber.get_current()

        # remove bias
        self.lerf_scores = self.lerf_scores - self.baseline_score
        self.morf_scores = self.morf_scores - self.baseline_score

        # calculate AUC
        lerf_auc = trapezoid(
            self.lerf_scores, dx=1. / float(len(self.lerf_scores)))
        morf_auc = trapezoid(
            self.morf_scores, dx=1. / float(len(self.morf_scores)))

        # deletion_img and insertion_img are final results, they are only
        # used for debug purpose
        return {
            "MoRF_scores": self.morf_scores,
            "LeRF_scores": self.lerf_scores,
            "MoRF_img": MoRF_img,
            "LeRF_img": LeRF_img,
            "LeRF_auc": lerf_auc,
            "MoRF_auc": morf_auc,
            "MoRF_img_history": self.img_history_morf,
            "LeRF_img_history": self.img_history_lerf
        }

    def _procedure_perturb(self,
                           perturber: GridPerturber,
                           num_pixels,
                           indices,
                           img_history=None):
        scores_after_perturb = [self.baseline_score.item()]
        replaced_pixels = 0
        # TODO to make it compatible with multi-label classification setting
        softmax = torch.nn.Softmax()
        while replaced_pixels < num_pixels:
            perturbed_imgs = []
            batch = min(num_pixels - replaced_pixels, self.batch)

            # perturb # of batch pixels
            for pixel in range(batch):
                perturb_index = (indices[0][replaced_pixels + pixel],
                                 indices[1][replaced_pixels + pixel])

                # perturb image using given pixels
                perturber.perturb(perturb_index[0], perturb_index[1])
                perturbed_imgs.append(perturber.get_current())
                if img_history is not None:
                    img_history.append(perturber.get_current())
            replaced_pixels += batch

            # get score after perturb
            device = next(self.model.parameters()).device
            perturbed_imgs = torch.stack(perturbed_imgs)
            score_after = softmax(self.model(
                perturbed_imgs.to(device)))[:, self.target]
            scores_after_perturb = np.concatenate(
                (scores_after_perturb, score_after.detach().cpu().numpy()))
        return scores_after_perturb
