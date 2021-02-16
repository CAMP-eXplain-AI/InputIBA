import numpy as np
import torch
import mmcv
import os.path as osp
from tqdm import tqdm
from iba.evaluation import EffectiveHeatRatios
from iba.datasets import build_dataset
import cv2
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Evaluate the heatmaps with EffectiveHeatRatios')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('file_name',
                        help='file name for saving the result file')
    parser.add_argument('--base-threshold',
                        type=float,
                        default=0.1,
                        help='base threshold of the metric')
    parser.add_argument('--roi',
                        type=str,
                        default='bboxes',
                        choices=['bboxes', 'masks'],
                        help='region of interest')
    parser.add_argument('--weight',
                        action='store_true',
                        help='weight the pixels by the heat')
    args = parser.parse_args()
    return args


def evaluate_ehr(cfg,
                 heatmap_dir,
                 work_dir,
                 file_name,
                 base_threshold=0.1,
                 roi='bboxes',
                 weight=True):
    mmcv.mkdir_or_exist(work_dir)

    dataset = build_dataset(cfg.data['val'])
    evaluator = EffectiveHeatRatios(base_threshold=base_threshold)
    assert roi in dataset[0].keys(
    ), f'dataset samples must contain the key: {roi}'

    res_dict = {}
    for i, sample in tqdm(enumerate(dataset)):
        img_name = sample['img_name']
        target = sample['target']
        roi_array = sample[roi]
        if isinstance(roi_array, torch.Tensor):
            # semantic masks
            roi_array = roi_array.numpy().astype(int)
        else:
            # bboxes
            roi_array = roi_array.astype(int)

        if not osp.exists(osp.join(heatmap_dir, img_name + '.png')):
            continue
        heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'),
                             cv2.IMREAD_UNCHANGED)
        res = evaluator.evaluate(heatmap, roi_array, weight_by_heat=weight)
        auc = res['auc']
        res_dict.update({img_name: {'target': target, 'auc': auc}})
    aucs = np.array([v['auc'] for v in res_dict.values()])
    print(f'auc: {aucs.mean():.5f} +/- {aucs.std():.5f}')
    file_name = osp.splitext(file_name)[0]
    file_path = osp.join(work_dir, file_name + '.json')
    mmcv.dump(res_dict, file=file_path)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    evaluate_ehr(cfg=cfg,
                 heatmap_dir=args.heatmap_dir,
                 work_dir=args.work_dir,
                 file_name=args.file_name,
                 base_threshold=args.base_threshold,
                 roi=args.roi,
                 weight=args.weight)


if __name__ == '__main__':
    main()
