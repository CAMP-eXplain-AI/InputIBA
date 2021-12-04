import numpy as np
import torch

from ..base import BaseEvaluation


class NLPSensitivityN(BaseEvaluation):

    def __init__(self, classifier, text_size, n, num_masks=100):
        """
        Initialize by generate random masks and indices of given text size (1D)
        Args:
            classifier: model to use
            text_size: size of the text (# of words)
            n:
            num_masks: number of random masks to genetate
        """
        self.classifier = classifier
        self.n = n
        self.device = next(self.classifier.parameters()).device
        self.indices, self.masks = self._generate_random_masks(
            num_masks, text_size, device=self.device)

    def evaluate(  # noqa
            self,
            heatmap: torch.Tensor,
            text: torch.Tensor,
            target: int,
            calculate_corr=False) -> dict:
        pertubated_text = []
        sum_attributions = []

        # compress heatmap to 1D if needed
        if heatmap.ndim == 2:
            heatmap = heatmap.mean(0)

        # get word embedding
        text_embedding = self.classifier.forward_embedding_only(text)
        for mask in self.masks:
            # perturb is done by interpolation, we perturb with zero word embedding
            pertubated_text.append(text_embedding * (1 - mask.unsqueeze(1)))
            sum_attributions.append((heatmap * mask).sum())
        sum_attributions = torch.stack(sum_attributions)
        input_text = pertubated_text + [text_embedding]
        with torch.no_grad():
            input_text = torch.stack(input_text, dim=1).to(self.device)
            text_lengths = torch.tensor([input_text.shape[0]
                                         ]).expand(input_text.shape[1])
            output = self.classifier.forward_no_embedding(
                input_text, text_lengths)
        output_pertubated = output[:-1]
        output_clean = output[-1:]

        diff = -(output_clean[:, 0] -
                 output_pertubated[:, 0]) if target == 0 else (
                     output_clean[:, 0] - output_pertubated[:, 0])
        score_diffs = diff.cpu().numpy()
        sum_attributions = sum_attributions.cpu().numpy()

        # calculate correlation for single image if requested
        corrcoef = None
        if calculate_corr:
            corrcoef = np.corrcoef(sum_attributions.flatten(),
                                   score_diffs.flatten())
        return {
            "correlation": corrcoef,
            "score_diffs": score_diffs,
            "sum_attributions": sum_attributions
        }

    def _generate_random_masks(self, num_masks, text_size, device='cuda:0'):
        """
        generate random masks with n pixel set to zero
        Args:
            num_masks: number of masks
            n: number of perturbed pixels
        Returns:
            indices
            masks
        """
        indices = []
        masks = []
        for _ in range(num_masks):
            idxs = np.unravel_index(
                np.random.choice(text_size, self.n, replace=False), text_size)
            indices.append(idxs)
            mask = np.zeros(text_size)
            mask[idxs] = 1
            masks.append(torch.from_numpy(mask).to(torch.float32).to(device))
        return indices, masks
