import torch
import numpy as np

from iba.evaluation.base import Evaluation
from iba.evaluation.perturber import PixelPerturber


class InsertionDeletion(Evaluation):
    def __init__(self, model, target, batch=10):
        self.model = model
        self.target = target
        self.deletion_scores = []
        self.insertion_scores = []
        self.batch = batch

    def eval(self, hmap: torch.Tensor, image: torch.Tensor) -> dict:
        # compress heatmap to 2D if needed
        if hmap.ndim == 3:
            hmap = hmap.mean(0)

        # sort pixel in attribution
        num_pixels = torch.numel(hmap)
        _, indices = torch.topk(hmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), hmap.size())

        # apply deletion game
        print("deletion game")
        deletion_perturber = PixelPerturber(image, torch.zeros_like(image))
        self.deletion_scores = self._procedure_perturb(deletion_perturber, num_pixels, indices)

        # apply insertion game
        print("insertion game")
        insertion_perturber = PixelPerturber(torch.zeros_like(image), image)
        self.insertion_scores = self._procedure_perturb(insertion_perturber, num_pixels, indices)

        return {"deletion_scores": self.deletion_scores, "insertion_scores": self.insertion_scores,
         "deletion_img": deletion_perturber.get_current(), "insertion_img": insertion_perturber.get_current()}

    def _procedure_perturb(self, perturber:PixelPerturber, num_pixels, indices):
        scores_after_perturb = []
        replaced_pixels = 0
        softmax = torch.nn.Softmax()
        while replaced_pixels < num_pixels:
            batch = min(num_pixels - replaced_pixels, self.batch)
            for pixel in range(batch):
                perturb_index = (indices[0][replaced_pixels + pixel], indices[1][replaced_pixels + pixel])

                # perturb img using given pixels
                perturber.perturb(perturb_index[0], perturb_index[1])

            # get score after perturb
            device = next(self.model.parameters()).device
            score_after = softmax(self.model(perturber.get_current().expand(1, -1, -1, -1).to(device)))[:, self.target]
            replaced_pixels += batch
            scores_after_perturb.append(score_after.detach().cpu().numpy().item())
        return scores_after_perturb
