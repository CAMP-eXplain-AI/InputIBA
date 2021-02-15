import torch
import numpy as np
from iba.evaluation.base import BaseEvaluation


class SensitivityN(BaseEvaluation):
    def __init__(self, model, target, image_size, n, num_masks=100):
        self.model = model
        self.target = target
        self.n = n
        self.dev = next(self.model.parameters()).device
        self.indices, self.masks = self._generate_random_masks(num_masks, image_size)

    def evaluate(self, heatmap: torch.Tensor, image: torch.Tensor, calculate_corr=True) -> dict:
        pertubated_imgs = []
        sum_attributions = []
        for mask in self.masks:
            # perturb is done by interpolation
            pertubated_imgs.append(image * (1 - mask))
            sum_attributions.append((heatmap * mask).sum())
        sum_attributions = torch.stack(sum_attributions)
        input_images = pertubated_imgs + [image]
        with torch.no_grad():
            input_images = torch.stack(input_images).to(self.dev)
            output = self.model(input_images)
        output_pertubated = output[:-1]
        output_clean = output[-1:]

        diff = output_clean[:, self.target] - output_pertubated[:, self.target]
        score_diffs = diff.cpu().numpy()
        sum_attributions = sum_attributions.cpu().numpy()

        # calculate correlation for single image if requested
        corrcoef = None
        if calculate_corr:
            corrcoef = np.corrcoef(sum_attributions.flatten(), score_diffs.flatten())
        return {"correlation": corrcoef, "score_diffs": score_diffs, "sum_attributions": sum_attributions}

    def evaluate_dataset(self, dataloader, heatmaps):
        pass

    def _generate_random_masks(self, num_masks, image_size):
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
        h, w = image_size
        for _ in range(num_masks):
            idxs = np.unravel_index(np.random.choice(h * w, self.n), (h, w))
            indices.append(idxs)
            mask = np.zeros((h, w))
            mask[idxs] = 1
            masks.append(torch.from_numpy(mask).float())
        return indices, masks
