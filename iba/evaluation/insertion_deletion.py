import torch
import torchvision
import numpy as np
from scipy.integrate import trapezoid

from iba.evaluation.base import BaseEvaluation
from iba.evaluation.perturber import PixelPerturber


class InsertionDeletion(BaseEvaluation):
    def __init__(self, attributer, pixel_batch_size=10, sigma=5.):
        self.attributer = attributer
        self.pixel_batch_size = pixel_batch_size
        self.gaussian_blurr = torchvision.transforms.GaussianBlur(int(2 * sigma - 1), sigma)

    @torch.no_grad()
    def evaluate(self,
                 heatmap,
                 img,
                 target): # noqa
        """# TODO to add docs

        Args:
            heatmap (Tensor): heatmap with shape (H, W) or (3, H, W).
            img (Tensor): image with shape (3, H, W).
            target (int): class index of the image.

        Returns:
            dict[str, Union[Tensor, np.array, float]]: a dictionary containing following fields
                - del_scores: ndarray,
                - ins_scores:
                - del_img:
                - ins_img:
                - ins_auc:
                - del_auc:
        """
        self.attributer.classifier.eval()

        # compress heatmap to 2D if needed
        if heatmap.ndim == 3:
            heatmap = heatmap.mean(0)

        # sort pixel in attribution
        num_pixels = torch.numel(heatmap)
        _, indices = torch.topk(heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), heatmap.size())

        # apply deletion game
        deletion_perturber = PixelPerturber(img, torch.zeros_like(img))
        deletion_scores = self._procedure_perturb(deletion_perturber, num_pixels, indices, target)

        # apply insertion game
        blurred_img = self.gaussian_blurr(img)
        insertion_perturber = PixelPerturber(blurred_img, img)
        insertion_scores = self._procedure_perturb(insertion_perturber, num_pixels, indices, target)

        # calculate AUC
        insertion_auc = trapezoid(insertion_scores, dx=1./len(insertion_scores))
        deletion_auc = trapezoid(deletion_scores, dx=1./len(deletion_scores))

        # deletion_img and insertion_img are final results, they are only used for debug purpose
        # TODO check if it is necessary to convert the Tensors to np.ndarray
        return {"del_scores": deletion_scores, "ins_scores": insertion_scores,
         "del_img": deletion_perturber.get_current(), "ins_img": insertion_perturber.get_current(),
         "ins_auc":insertion_auc, "del_auc":deletion_auc}

    def _procedure_perturb(self,
                           perturber,
                           num_pixels,
                           indices,
                           target):
        """ # TODO to add docs

        Args:
            perturber (PixelPerturber):
            num_pixels (int):
            indices (tuple):
            target (int):

        Returns:
            np.ndarray:
        """
        scores_after_perturb = []
        replaced_pixels = 0
        while replaced_pixels < num_pixels:
            perturbed_imgs = []
            for i in range(8):
                batch = min(num_pixels - replaced_pixels, self.pixel_batch_size)

                # perturb # of pixel_batch_size pixels
                for pixel in range(batch):
                    perturb_index = (indices[0][replaced_pixels + pixel], indices[1][replaced_pixels + pixel])

                    # perturb img using given pixels
                    perturber.perturb(perturb_index[0], perturb_index[1])
                perturbed_imgs.append(perturber.get_current())
                replaced_pixels += batch
                if replaced_pixels == num_pixels:
                    break

            # get score after perturb
            device = next(self.attributer.classifier.parameters()).device
            perturbed_imgs = torch.stack(perturbed_imgs)
            logits = self.attributer.classifier(perturbed_imgs.to(device))
            score_after = torch.softmax(logits, dim=-1)[:, target]
            scores_after_perturb = np.concatenate((scores_after_perturb, score_after.detach().cpu().numpy()))
        return scores_after_perturb
