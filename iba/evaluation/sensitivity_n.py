import torch
import numpy as np
from iba.evaluation.base import BaseEvaluation


class SensitivityN(BaseEvaluation):

    def __init__(self, classifier, img_size, n, num_masks=100):
        self.classifier = classifier
        self.n = n
        self.device = next(self.classifier.parameters()).device
        self.indices, self.masks = self._generate_random_masks(
            num_masks, img_size, device=self.device)

    def evaluate(  # noqa
            self,
            heatmap: torch.Tensor,
            img: torch.Tensor,
            target: int,
            calculate_corr=False) -> dict:
        pertubated_imgs = []
        sum_attributions = []
        for mask in self.masks:
            # perturb is done by interpolation
            pertubated_imgs.append(img * (1 - mask))
            sum_attributions.append((heatmap * mask).sum())
        sum_attributions = torch.stack(sum_attributions)
        input_imgs = pertubated_imgs + [img]
        with torch.no_grad():
            input_imgs = torch.stack(input_imgs).to(self.device)
            output = self.classifier(input_imgs)
        output_pertubated = output[:-1]
        output_clean = output[-1:]

        diff = output_clean[:, target] - output_pertubated[:, target]
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

    def _generate_random_masks(self, num_masks, img_size, device='cuda:0'):
        """
        generate random masks with n pixel set to zero
        Args:
            num_masks: number of masks
            n: number of perturbed pixels
        Returns:
            masks
        """
        indices = []
        masks = []
        h, w = img_size
        for _ in range(num_masks):
            idxs = np.unravel_index(
                np.random.choice(h * w, self.n, replace=False), (h, w))
            indices.append(idxs)
            mask = np.zeros((h, w))
            mask[idxs] = 1
            masks.append(torch.from_numpy(mask).to(torch.float32).to(device))
        return indices, masks
