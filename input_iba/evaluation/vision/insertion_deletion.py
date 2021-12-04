import numpy as np
import torch
from scipy.integrate import trapezoid
from torchvision.transforms import GaussianBlur

from ..base import BaseEvaluation
from ..perturber import PixelPerturber


class VisionInsertionDeletion(BaseEvaluation):

    def __init__(self, classifier, pixel_batch_size=10, sigma=5.):
        self.classifier = classifier
        self.classifier.eval()
        self.pixel_batch_size = pixel_batch_size
        self.gaussian_blurr = GaussianBlur(int(2 * sigma - 1), sigma)

    @torch.no_grad()
    def evaluate(self, heatmap, input_tensor, target):  # noqa
        """# TODO to add docs

        Args:
            heatmap (Tensor): heatmap with shape (H, W) or (3, H, W).
            input_tensor (Tensor): image with shape (3, H, W).
            target (int): class index of the image.

        Returns:
            dict[str, Union[Tensor, np.array, float]]: a dictionary
                containing following fields:
                - del_scores: ndarray,
                - ins_scores:
                - del_input:
                - ins_input:
                - ins_auc:
                - del_auc:
        """

        # sort pixel in attribution
        num_pixels = torch.numel(heatmap)
        _, indices = torch.topk(heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), heatmap.size())

        # apply deletion game
        deletion_perturber = PixelPerturber(input_tensor,
                                            torch.zeros_like(input_tensor))
        deletion_scores = self._procedure_perturb(deletion_perturber,
                                                  num_pixels, indices, target)

        # apply insertion game
        blurred_input = self.gaussian_blurr(input_tensor)
        insertion_perturber = PixelPerturber(blurred_input, input_tensor)
        insertion_scores = self._procedure_perturb(insertion_perturber,
                                                   num_pixels, indices, target)

        # calculate AUC
        insertion_auc = trapezoid(
            insertion_scores, dx=1. / len(insertion_scores))
        deletion_auc = trapezoid(deletion_scores, dx=1. / len(deletion_scores))

        # deletion_input and insertion_input are final results, they are
        # only used for debug purpose
        # TODO check if it is necessary to convert the Tensors to np.ndarray
        return {
            "del_scores": deletion_scores,
            "ins_scores": insertion_scores,
            "del_input": deletion_perturber.get_current(),
            "ins_input": insertion_perturber.get_current(),
            "ins_auc": insertion_auc,
            "del_auc": deletion_auc
        }

    def _procedure_perturb(self, perturber, num_pixels, indices, target):
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
            perturbed_inputs = []
            for i in range(80):
                batch = min(num_pixels - replaced_pixels,
                            self.pixel_batch_size)

                # perturb # of pixel_batch_size pixels
                for pixel in range(batch):
                    perturb_index = (indices[0][replaced_pixels + pixel],
                                     indices[1][replaced_pixels + pixel])

                    # perturb input using given pixels
                    perturber.perturb(perturb_index[0], perturb_index[1])
                perturbed_inputs.append(perturber.get_current())
                replaced_pixels += batch
                if replaced_pixels == num_pixels:
                    break

            # get score after perturb
            device = next(self.classifier.parameters()).device
            perturbed_inputs = torch.stack(perturbed_inputs)
            logits = self.classifier(perturbed_inputs.to(device))
            score_after = torch.softmax(logits, dim=-1)[:, target]
            scores_after_perturb = np.concatenate(
                (scores_after_perturb, score_after.detach().cpu().numpy()))
        return scores_after_perturb
