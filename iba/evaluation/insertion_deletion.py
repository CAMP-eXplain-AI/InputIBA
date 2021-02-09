import torch
import torchvision
import numpy as np
from scipy.integrate import trapezoid

from iba.evaluation.base import BaseEvaluation
from iba.evaluation.perturber import PixelPerturber


class InsertionDeletion(BaseEvaluation):
    def __init__(self, model, target, batch=10, gaussian_sigma = 5.):
        self.model = model
        self.target = target
        self.deletion_scores = []
        self.insertion_scores = []
        self.batch = batch
        self.gaussian_blurr = torchvision.transforms.GaussianBlur(int(2*gaussian_sigma-1),gaussian_sigma)

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
        blurred_img = self.gaussian_blurr(image)
        insertion_perturber = PixelPerturber(blurred_img, image)
        self.insertion_scores = self._procedure_perturb(insertion_perturber, num_pixels, indices)

        # calculate AUC
        insertion_auc = trapezoid(self.insertion_scores, dx=1./float(len(self.insertion_scores)))
        deletion_auc = trapezoid(self.deletion_scores, dx=1./float(len(self.deletion_scores)))

        # deletion_img and insertion_img are final results, they are only used for debug purpose
        return {"deletion_scores": self.deletion_scores, "insertion_scores": self.insertion_scores,
         "deletion_img": deletion_perturber.get_current(), "insertion_img": insertion_perturber.get_current(), 
         "insertion_auc":insertion_auc, "deletion_auc":deletion_auc}

    def _procedure_perturb(self, perturber: PixelPerturber, num_pixels, indices):
      scores_after_perturb = []
      replaced_pixels = 0
      softmax = torch.nn.Softmax()
      while replaced_pixels < num_pixels:
          perturbed_imgs = []
          for i in range(8):
              batch = min(num_pixels - replaced_pixels, self.batch)

              # perturb # of batch pixels
              for pixel in range(batch):
                  perturb_index = (indices[0][replaced_pixels + pixel], indices[1][replaced_pixels + pixel])

                  # perturb image using given pixels
                  perturber.perturb(perturb_index[0], perturb_index[1])
              perturbed_imgs.append(perturber.get_current())
              replaced_pixels += batch
              if replaced_pixels == num_pixels:
                  break

          # get score after perturb
          device = next(self.model.parameters()).device
          perturbed_imgs = torch.stack(perturbed_imgs)
          score_after = softmax(self.model(perturbed_imgs.to(device)))[:, self.target]
          scores_after_perturb = np.concatenate((scores_after_perturb, score_after.detach().cpu().numpy()))
      return scores_after_perturb
