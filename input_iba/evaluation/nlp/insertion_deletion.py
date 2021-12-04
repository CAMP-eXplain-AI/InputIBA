import numpy as np
import torch
import torchvision
from scipy.integrate import trapezoid

from ..base import BaseEvaluation
from .perturber import WordPerturber


class NLPInsertionDeletion(BaseEvaluation):

    def __init__(self, classifier, pixel_batch_size=2, sigma=5.):
        self.classifier = classifier
        self.classifier.eval()
        self.pixel_batch_size = pixel_batch_size
        self.gaussian_blurr = torchvision.transforms.GaussianBlur(
            int(2 * sigma - 1), sigma)

    @torch.no_grad()
    def evaluate(self, heatmap, text, target):  # noqa
        """# TODO to add docs
        Args:
            heatmap (Tensor): heatmap with shape (H, W) or (3, H, W).
            text (Tensor): image with shape (3, H, W).
            target (int): class index of the image.
        Returns:
            dict[str, Union[Tensor, np.array, float]]: a dictionary containing following fields
                - del_scores: ndarray,
                - ins_scores:
                - del_text:
                - ins_text:
                - ins_auc:
                - del_auc:
        """
        # compress heatmap to 1D if needed
        if heatmap.ndim == 2:
            heatmap = heatmap.mean(0)

        # sort pixel in attribution
        num_pixels = torch.numel(heatmap)
        _, indices = torch.topk(heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), heatmap.size())

        # get word embedding
        text_embedding = self.classifier.forward_embedding_only(text)

        # apply deletion game
        deletion_perturber = WordPerturber(text_embedding,
                                           torch.zeros_like(text_embedding))
        deletion_scores = self._procedure_perturb(deletion_perturber,
                                                  int(num_pixels * 0.4),
                                                  indices, target)

        # apply insertion game
        insertion_perturber = WordPerturber(
            torch.zeros_like(text_embedding), text_embedding)
        insertion_scores = self._procedure_perturb(insertion_perturber,
                                                   int(num_pixels * 0.4),
                                                   indices, target)

        # calculate AUC
        insertion_auc = trapezoid(
            insertion_scores, dx=1. / len(insertion_scores))
        deletion_auc = trapezoid(deletion_scores, dx=1. / len(deletion_scores))

        # deletion_text and insertion_text are final results, they are only used for debug purpose
        # TODO check if it is necessary to convert the Tensors to np.ndarray
        return {
            "del_scores": deletion_scores,
            "ins_scores": insertion_scores,
            "del_text": deletion_perturber.get_current(),
            "ins_text": insertion_perturber.get_current(),
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
            perturbed_texts = []
            for i in range(32):
                batch = min(num_pixels - replaced_pixels,
                            self.pixel_batch_size)

                # perturb # of pixel_batch_size pixels
                for pixel in range(batch):
                    perturb_index = indices[0][replaced_pixels + pixel]

                    # perturb text using given pixels
                    perturber.perturb(perturb_index)
                perturbed_texts.append(perturber.get_current())
                replaced_pixels += batch
                if replaced_pixels == num_pixels:
                    break

            # get score after perturb
            device = next(self.classifier.parameters()).device

            # stack at dim 1 (due to NLP)
            perturbed_texts = torch.stack(perturbed_texts, dim=1)
            text_lengths = torch.tensor([perturbed_texts.shape[0]
                                         ]).expand(perturbed_texts.shape[1])
            logits = self.classifier.forward_no_embedding(
                perturbed_texts.to(device), text_lengths)
            score_after = torch.sigmoid(
                logits) if target else 1 - torch.sigmoid(logits)
            scores_after_perturb = np.concatenate(
                (scores_after_perturb,
                 score_after.squeeze(1).detach().cpu().numpy()))
        return scores_after_perturb
