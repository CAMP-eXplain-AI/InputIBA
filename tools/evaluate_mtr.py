import numpy as np
import torch
import mmcv
import os.path as osp
from tqdm import tqdm
from iba.evaluation import MultiThresholdRatios
from iba.datasets import build_dataset
import cv2
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Evaluate the heatmaps with MultiThresholdRatios')
    parser.add_argument('config',
                        help='config file of the attribution method')
    parser.add_argument('heatmap_dir',
                        help='directory of the heatmaps')
    parser.add_argument('work_dir',
                        help='directory to save the result file')
    parser.add_argument('file_name',
                        help='file name for saving the result file')
    parser.add_argument('--base-threshold',
                        type=float,
                        default=0.1,
                        help='base threshold of the metric')
    parser.add_argument('--roi',
                        type=str,
                        choices=['bbox', 'mask'],
                        help='region of interest')
    args = parser.parse_args()
    return args


def evaluate_mtr(cfg,
                 heatmap_dir,
                 work_dir,
                 file_name,
                 base_threshold=0.1,
                 roi='bbox'):
    mmcv.mkdir_or_exist(work_dir)

    dataset = build_dataset(cfg.data['val'])
    evaluator = MultiThresholdRatios(base_threshold=base_threshold)
    assert roi in dataset[0].keys(), f'dataset samples must contain the key: {roi}'

    res_dict = {}
    for sample in tqdm(dataset):
        img_name = sample['img_name']
        target = sample['target']
        roi = sample[roi]
        if isinstance(roi, torch.Tensor):
            # semantic masks
            roi = roi.numpy().astype(int)
        else:
            # bboxes
            roi = roi.astype(int)

        if not osp.exists(osp.join(heatmap_dir, img_name + '.png')):
            continue
        heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'), cv2.IMREAD_UNCHANGED)
        res = evaluator.evaluate(heatmap, roi)
        auc = res['auc']
        res_dict.update({'img_name': img_name,
                         'target': target,
                         'auc': auc})

    file_name = osp.splitext(file_name)[0]
    file_path = osp.join(work_dir, file_name + '.json')
    mmcv.dump(res_dict, file=file_path)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    evaluate_mtr(cfg=cfg,
                 heatmap_dir=args.heatmap_dir,
                 work_dir=args.work_dir,
                 file_name=args.file_name,
                 base_threshold=args.base_threshold,
                 roi=args.roi)


if __name__ == '__main__':
    main()