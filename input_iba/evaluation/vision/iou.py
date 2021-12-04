import numpy as np

from ..base import BaseEvaluation


class IoU(BaseEvaluation):

    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0, 1.0, 0.02)
        self.thresholds = thresholds

    def evaluate(self, heatmap, gt_mask):
        assert gt_mask.max(
        ) <= 1 and gt_mask.ndim == 1, "gt_mask should be a binary mask"
        gt_mask = gt_mask.astype(bool)
        ious = []
        for t in self.thresholds:
            bin_heatmap = heatmap > t
            iou_ = self.iou(bin_heatmap, gt_mask)
            ious.append(iou_)
        return dict(thresholds=self.thresholds.tolist(), ious=ious)

    @staticmethod
    def iou(heatmap, gt_mask, eps=1e-6):
        inter = (heatmap * gt_mask).sum()
        union = np.logical_or(heatmap, gt_mask).sum()
        return inter / (union + eps)
