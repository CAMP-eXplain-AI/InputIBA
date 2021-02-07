from iba.evaluation import MultiThresholdRatios
import numpy as np
from time import time


class TestMultiThresholdRatios:
    def test_case_0(self):
        h, w = 500, 500
        bbox = np.array([100, 100, 350, 350])
        heatmap = np.zeros((h, w))
        heatmap[bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1.0
        metric = MultiThresholdRatios(base_threshold=0.1)
        ts = time()
        res = metric.evaluate(heatmap, bbox)
        print(f'time elapsed: {time() - ts}')
        assert 1.0 - res['auc'] < 0.01

    def test_case_1(self):
        h, w = 500, 500
        bbox = np.array([100, 100, 350, 350])
        heatmap = np.zeros((h, w))
        heatmap[bbox[1]: bbox[3] - 50, bbox[0]: bbox[2] - 50] = 1.0
        metric = MultiThresholdRatios(base_threshold=0.1)
        ts = time()
        res = metric.evaluate(heatmap, bbox)
        print(f'time elapsed: {time() - ts}')
        assert 1.0 - res['auc'] < 0.01

    def test_case_2(self):
        h, w= 100, 100
        bbox = np.array([40, 40, 60, 60])
        heatmap = np.zeros((h, w))
        heatmap[bbox[1] - 20: bbox[3] + 20, bbox[0] - 20: bbox[2] + 20] = 0.90
        heatmap[bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1.0
        metric = MultiThresholdRatios(base_threshold=0.1)
        ts = time()
        res = metric.evaluate(heatmap, bbox)
        print(f'time elapsed: {time() - ts}')
        print(f"auc: {res['auc']}")
        assert res['auc'] < 0.8