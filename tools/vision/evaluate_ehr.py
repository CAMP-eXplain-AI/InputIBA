import numpy as np
import torch
import mmcv
import os.path as osp
from tqdm import tqdm
from iba.evaluation import EffectiveHeatRatios
from iba.datasets import build_dataset
from iba.utils import get_valid_set
import cv2
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Evaluate the heatmaps with EffectiveHeatRatios')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument(
        'file_name', help='file name for saving the result file')
    parser.add_argument(
        '--weight', action='store_true', help='weight the pixels by the heat')
    parser.add_argument(
        '--scores-file',
        help='File that records the predicted probability of corresponding '
        'target class')
    parser.add_argument(
        '--scores-threshold',
        type=float,
        default=0.6,
        help='Threshold for filtering the samples with low predicted target '
        'probabilities')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=0,
        help='Number of samples to check, 0 means checking all the (filtered) '
        'samples')
    parser.add_argument(
        '--base-threshold',
        type=float,
        default=0.1,
        help='base threshold of the metric')
    parser.add_argument(
        '--roi',
        type=str,
        default='bboxes',
        choices=['bboxes', 'masks'],
        help='region of interest')
    args = parser.parse_args()
    return args


def evaluate_ehr(cfg,
                 heatmap_dir,
                 work_dir,
                 file_name,
                 scores_file=None,
                 scores_threshold=0.6,
                 num_samples=0,
                 base_threshold=0.1,
                 roi='bboxes',
                 weight=True):
    mmcv.mkdir_or_exist(work_dir)

    attr_set = build_dataset(cfg.data['attribution'])
    attr_set = get_valid_set(
        attr_set,
        scores_file=scores_file,
        scores_threshold=scores_threshold,
        num_samples=num_samples)

    evaluator = EffectiveHeatRatios(base_threshold=base_threshold)
    assert roi in attr_set[0].keys(
    ), f'dataset samples must contain the key: {roi}'

    res_dict = {}
    for i, sample in tqdm(enumerate(attr_set)):
        input_name = sample['input_name']
        target = sample['target']
        roi_array = sample[roi]
        if isinstance(roi_array, torch.Tensor):
            # semantic masks
            roi_array = roi_array.numpy().astype(int)
        else:
            # bboxes
            roi_array = roi_array.astype(int)

        if not osp.exists(osp.join(heatmap_dir, input_name + '.png')):
            continue

        heatmap_path = osp.join(heatmap_dir, input_name + '.png')
        assert osp.exists(heatmap_path)
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)

        # compute the ratio of roi_area / image_size
        roi_mask = np.zeros_like(heatmap)
        if roi_array.ndim == 1:
            roi_array = roi_array[None, :]
        if roi_array.shape[-1] == 4:
            # bbox
            for roi_single in roi_array:
                x1, y1, x2, y2 = roi_single
                roi_mask[y1:y2, x1:x2] = 1
            roi_area_ratio = roi_mask.sum() / (
                roi_mask.shape[-1] * roi_mask.shape[-2])
        else:
            # binary mask
            roi_area_ratio = roi_array.sum() / (
                roi_array.shape[-1] * roi_mask.shape[-2])
        # only select the samples for which the roi area ratio is smaller
        # than a threshold
        if roi_area_ratio > 0.33:
            continue

        res = evaluator.evaluate(heatmap, roi_array, weight_by_heat=weight)
        auc = res['auc']
        res_dict.update({input_name: {'target': target, 'auc': auc}})
    aucs = np.array([v['auc'] for v in res_dict.values()])
    print(f'auc: {aucs.mean():.5f} +/- {aucs.std():.5f}')
    file_name = osp.splitext(file_name)[0]
    file_path = osp.join(work_dir, file_name + '.json')
    mmcv.dump(res_dict, file=file_path)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    evaluate_ehr(
        cfg=cfg,
        heatmap_dir=args.heatmap_dir,
        work_dir=args.work_dir,
        file_name=args.file_name,
        scores_file=args.scores_file,
        scores_threshold=args.scores_threshold,
        num_samples=args.num_samples,
        base_threshold=args.base_threshold,
        roi=args.roi,
        weight=args.weight)


if __name__ == '__main__':
    main()
