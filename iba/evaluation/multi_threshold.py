import numpy as np
from scipy.integrate import trapezoid
from .base import BaseEvaluation


class MultiThresholdRatios(BaseEvaluation):
    def __init__(self, base_threshold=0.1):
        self.base_threshold = base_threshold
        self.quantiles = np.linspace(0, 1.0, 11)

    def evaluate(self, heatmap, roi):   # noqa
        if heatmap.ndim == 3:
            heatmap = heatmap.mean(0)
        roi_mask = np.zeros_like(heatmap).astype(bool)
        if isinstance(roi, list):
            roi = np.array(roi)
        if roi.shape[-1] == 4:
            if roi.ndim == 1:
                roi = roi[None, :]
            for bbox in roi:
                x1, y1, x2, y2 = bbox
                roi_mask[y1:y2, x1:x2] = 1
        else:
            assert roi.shape == heatmap.shape
            roi_mask = roi.astype(bool)

        if heatmap.max() > 1.0:
            heatmap = heatmap.astype(float) / 255.0
        poi_mask = heatmap >= self.base_threshold
        heatmap[~poi_mask] = 0.0
        poi = heatmap[poi_mask]
        ratios = np.zeros(len(self.quantiles))

        max_heat = heatmap.max()
        for i, q in enumerate(self.quantiles):
            t = min(np.quantile(poi, q), max_heat * 0.99)

            bin_mask = heatmap >= t
            total_points = bin_mask.sum()
            heat_in_roi = heatmap[bin_mask * roi_mask].sum()
            print(f'{i}, t: {t}, '
                  f'total_points: {total_points}, '
                  f'heat_in_roi: {heat_in_roi}')
            ratio = heat_in_roi / (total_points + 1e-8)
            ratios[i] = ratio
        print(ratios)
        val = self.integrate(ratios, self.quantiles)
        return dict(auc=val)

    @staticmethod
    def integrate(ratios, quantiles):
        return trapezoid(ratios, quantiles)