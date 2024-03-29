import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from argparse import ArgumentParser
from copy import deepcopy
from mmcv.runner.utils import set_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from input_iba.datasets import build_dataset
from input_iba.evaluation import VisionSensitivityN
from input_iba.models import build_classifiers
from input_iba.utils import get_valid_set


def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('file_name', help='file name fo saving the results')
    parser.add_argument(
        '--scores-file',
        help='File that records the predicted probability of '
        'corresponding target class')
    parser.add_argument(
        '--scores-threshold',
        type=float,
        default=0.6,
        help='Threshold for filtering the samples with low predicted target '
        'probabilities')
    parser.add_argument(
        '--log-n-max',
        type=float,
        default=4.5,
        help='maximal N of Sensitivity-N')
    parser.add_argument(
        '--log-n-ticks',
        type=float,
        default=0.1,
        help='Ticks for determining the Ns')
    parser.add_argument(
        '--num-masks',
        type=int,
        default=100,
        help='Number of random masks of Sensitivity-N')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=0,
        help='Number of samples to evaluate, 0 means checking all the samples')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    args = parser.parse_args()
    return args


def sensitivity_n(cfg,
                  heatmap_dir,
                  work_dir,
                  file_name,
                  scores_file=None,
                  scores_threshold=0.6,
                  log_n_max=4.5,
                  log_n_ticks=0.1,
                  num_masks=100,
                  num_samples=0,
                  device='cuda:0'):
    logger = mmcv.get_logger('iba')
    mmcv.mkdir_or_exist(work_dir)
    attr_set = build_dataset(cfg.data['attribution'])
    # check if n is valid
    input_h, input_w = attr_set[0]['input'].shape[-2:]
    max_allowed_n = np.log(input_h * input_w)
    assert log_n_max < max_allowed_n, \
        f"log_n_max must smaller than {max_allowed_n}, but got {log_n_max}"

    attr_set = get_valid_set(
        attr_set,
        scores_file=scores_file,
        scores_threshold=scores_threshold,
        num_samples=num_samples)

    attr_loader_cfg = deepcopy(cfg.data['data_loader'])
    attr_loader_cfg.update({'shuffle': False})
    attr_loader = DataLoader(attr_set, **attr_loader_cfg)
    classifier = build_classifiers(cfg.attributor['classifier']).to(device)

    sample = attr_set[0]['input']
    h, w = sample.shape[1:]
    results = {}

    try:
        n_list = np.logspace(
            0, log_n_max, int(log_n_max / log_n_ticks), base=10.0, dtype=int)
        # to eliminate the duplicate elements caused by rounding
        n_list = np.unique(n_list)
        logger.info(f"n_list: [{', '.join(map(str,n_list))}]")
        pbar = tqdm(total=len(n_list) * len(attr_loader))
        for n in n_list:
            evaluator = VisionSensitivityN(
                classifier, input_size=(h, w), n=n, num_masks=num_masks)

            corr_all = []
            for batch in attr_loader:
                inputs = batch['input']
                targets = batch['target']
                input_names = batch['input_name']

                for input_tensor, target, input_name in zip(
                        inputs, targets, input_names):
                    input_tensor = input_tensor.to(device)
                    target = target.item()

                    heatmap_path = osp.join(heatmap_dir, input_name + '.png')
                    assert osp.exists(heatmap_path), \
                        f'File {heatmap_path} does not exist or is empty'
                    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
                    heatmap = torch.from_numpy(heatmap).to(
                        input_tensor) / 255.0

                    res_single = evaluator.evaluate(
                        heatmap, input_tensor, target, calculate_corr=True)
                    corr = res_single['correlation'][1, 0]
                    corr_all.append(corr)
                pbar.update(1)
            results.update({int(n): np.mean(corr_all)})
    except KeyboardInterrupt as e:
        logger.info(f'Evaluation ended due to KeyboardInterrupt')
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return
    except AssertionError as e:
        logger.info(f'Evaluation ended due to {e}')
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return
    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)
    sensitivity_n(
        cfg=cfg,
        heatmap_dir=args.heatmap_dir,
        work_dir=args.work_dir,
        file_name=args.file_name,
        scores_file=args.scores_file,
        scores_threshold=args.scores_threshold,
        log_n_max=args.log_n_max,
        log_n_ticks=args.log_n_ticks,
        num_masks=args.num_masks,
        num_samples=args.num_samples,
        device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
