import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from argparse import ArgumentParser
from mmcv.runner.utils import set_random_seed
from tqdm import tqdm

from input_iba.datasets import build_dataset
from input_iba.evaluation.nlp.sensitivity_n import NLPSensitivityN
from input_iba.models import build_classifiers


def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation on IMDB')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('file_name', help='file name fo saving the results')
    parser.add_argument(
        '--num-masks',
        type=int,
        default=100,
        help='Number of random masks of Sensitivity-N')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2000,
        help='Number of samples to evaluate, 0 means checking all the samples')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    args = parser.parse_args()
    return args


def sensitivity_n(cfg,
                  heatmap_dir,
                  work_dir,
                  file_name,
                  num_masks=100,
                  num_samples=0,
                  device='cuda:0'):
    logger = mmcv.get_logger('iba')
    mmcv.mkdir_or_exist(work_dir)

    classifier = build_classifiers(cfg.attributor['classifier']).to(device)
    dataset = build_dataset(cfg.data['attribution'])

    results = {}

    try:
        n_list = np.linspace(0.1, 1, num=10)
        # to eliminate the duplicate elements caused by rounding
        n_list = np.unique(n_list)
        logger.info(f"n_list: [{', '.join(map(str,n_list))}]")
        pbar = tqdm(total=len(n_list) * num_samples)
        for n in n_list:
            count = 0

            corr_all = []
            for datapoint in tqdm(dataset):
                count += 1
                input_tensor = datapoint['input'].to(device)
                target = datapoint['target']
                input_name = datapoint['input_name']

                heatmap_path = osp.join(heatmap_dir, input_name + '.png')
                assert osp.exists(heatmap_path), \
                    f'File {heatmap_path} does not exist or is empty'
                heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
                heatmap = torch.from_numpy(heatmap).to(device) / 255.0

                evaluator = NLPSensitivityN(
                    classifier,
                    input_tensor.shape[0],
                    n=int(input_tensor.shape[0] * n),
                    num_masks=num_masks)
                res_single = evaluator.evaluate(
                    heatmap, input_tensor, target, calculate_corr=True)
                corr = res_single['correlation'][1, 0]
                corr_all.append(corr)
                pbar.update(1)
                if count >= num_samples:
                    break
            results.update({n: np.mean(corr_all)})
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
        num_masks=args.num_masks,
        num_samples=args.num_samples,
        device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
