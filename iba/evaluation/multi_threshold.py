import numpy as np
from scipy.integrate import trapezoid


class MultiThresholdMetric:
    def __init__(self, base_threshold=0.1):
        self.base_threshold = base_threshold
        self.quantiles = np.arange(0.0, 1.1, 0.1)

    def evaluate(self, heatmap, roi):
        # TODO measure the time consumption
        # TODO use multiprocessing
        assert heatmap.ndim == 2
        roi_mask = np.zeros_like(heatmap).astype(bool)
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
        heatmap[heatmap < self.base_threshold] = 0.0

        ratios = np.zeros(len(self.quantiles))

        for i, q in enumerate(self.quantiles):
            t = np.quantile(heatmap, q)
            bin_mask = heatmap >= t
            num_points = bin_mask.sum()

            num_points_in_roi = (bin_mask * roi_mask).sum()
            ratio = num_points_in_roi / num_points
            ratios[i] = ratio

        val = self.integrate(ratios, self.quantiles)
        return val

    @staticmethod
    def integrate(ratios, quantiles):
        return trapezoid(ratios, quantiles)