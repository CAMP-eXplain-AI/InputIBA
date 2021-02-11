import torch
import torchvision
import numpy as np
from scipy.integrate import trapezoid

from iba.evaluation.base import BaseEvaluation
from iba.evaluation.perturber import GridView, GridPerturber


class Degradation(BaseEvaluation):
    def __init__(self, model, target, tile_size, store_imgs=False, batch=8):
        self.model = model
        self.batch = batch
        self.target = target
        self.MoRF_scores = []
        self.LeRF_scores = []
        self.tile_size = tile_size
        self.img_history_MoRF = [] if store_imgs else None
        self.img_history_LeRF = [] if store_imgs else None

    def eval(self, hmap: torch.Tensor, image: torch.Tensor) -> dict:
        self.model.eval()

        # compress heatmap to 2D if needed
        if hmap.ndim == 3:
            hmap = hmap.mean(0)

        # construct perturbed img and get baseline score
        # TODO implement

        # get 2d tile attribution
        perturber = GridPerturber(image, torch.zeros_like(image), self.tile_size)
        grid_heatmap = torch.zeros(perturber.get_grid_shape())
        for r in range(grid_heatmap.shape[0]):
            for c in range(grid_heatmap.shape[1]):
                grid_heatmap[r][c] = hmap[perturber.view.tile_slice(r, c)].sum()

        # sort tile in attribution
        num_pixels = torch.numel(grid_heatmap)
        _, indices = torch.topk(grid_heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), hmap.size())
        _, reverse_indices = torch.topk(grid_heatmap.flatten(), num_pixels, largest=False)
        reverse_indices = np.unravel_index(reverse_indices.cpu().numpy(), hmap.size())

        # apply deletion game using MoRF
        print("MoRF deletion")
        self.deletion_scores = self._procedure_perturb(perturber, num_pixels, indices, self.img_history_MoRF)
        MoRF_img = perturber.get_current()

        # apply deletion game using LeRF
        perturber = GridPerturber(image, torch.zeros_like(image), self.tile_size)
        print("LeRF deletion")
        self.insertion_scores = self._procedure_perturb(perturber, num_pixels, reverse_indices, self.img_history_LeRF)
        LeRF_img = perturber.get_current()

        # calculate AUC
        insertion_auc = trapezoid(self.insertion_scores, dx=1. / float(len(self.insertion_scores)))
        deletion_auc = trapezoid(self.deletion_scores, dx=1. / float(len(self.deletion_scores)))

        # deletion_img and insertion_img are final results, they are only used for debug purpose
        return {"deletion_scores": self.deletion_scores, "insertion_scores": self.insertion_scores,
                "MoRF_img": MoRF_img, "LeRF_img": LeRF_img,
                "insertion_auc": insertion_auc, "deletion_auc": deletion_auc,
                "MoRF_img_history":self.img_history_MoRF, "LeRF_img_history":self.img_history_LeRF}

    def _procedure_perturb(self, perturber: GridPerturber, num_pixels, indices, img_history=None):
        scores_after_perturb = []
        replaced_pixels = 0
        softmax = torch.nn.Softmax()
        while replaced_pixels < num_pixels:
            perturbed_imgs = []
            batch = min(num_pixels - replaced_pixels, self.batch)

            # perturb # of batch pixels
            for pixel in range(batch):
                perturb_index = (indices[0][replaced_pixels + pixel], indices[1][replaced_pixels + pixel])

                # perturb image using given pixels
                perturber.perturb(perturb_index[0], perturb_index[1])
                perturbed_imgs.append(perturber.get_current())
                if img_history is not None:
                    img_history.append(perturber.get_current())
            replaced_pixels += batch

            # get score after perturb
            device = next(self.model.parameters()).device
            perturbed_imgs = torch.stack(perturbed_imgs)
            score_after = softmax(self.model(perturbed_imgs.to(device)))[:, self.target]
            scores_after_perturb = np.concatenate((scores_after_perturb, score_after.detach().cpu().numpy()))
        return scores_after_perturb
