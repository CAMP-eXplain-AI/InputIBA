import os.path as osp

import cv2
import gc
import mmcv
from argparse import ArgumentParser
from copy import deepcopy
from mmcv.runner.utils import set_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from input_iba.datasets import build_dataset
from input_iba.evaluation import SanityCheck
from input_iba.models import build_attributor
from input_iba.utils import get_valid_set


def parse_args():
    parser = ArgumentParser('Sanity check')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('file_name', help='file name fo saving the results')
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
        help='Number of samples to check, 0 means checking all the samples')
    parser.add_argument(
        '--save-heatmaps',
        action='store_true',
        default=False,
        help='Whether to save the heatmaps produced by the perturbed models')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    args = parser.parse_args()
    return args


def sanity_check(cfg,
                 heatmap_dir,
                 work_dir,
                 file_name,
                 scores_file=None,
                 scores_threshold=0.6,
                 num_samples=0,
                 save_heatmaps=False,
                 device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)
    est_set = build_dataset(cfg.data['estimation'])
    attr_set = build_dataset(cfg.data['attribution'])
    attr_set = get_valid_set(
        attr_set,
        scores_file=scores_file,
        scores_threshold=scores_threshold,
        num_samples=num_samples)

    est_loader = DataLoader(est_set, **cfg.data['data_loader'])
    attr_loader_cfg = deepcopy(cfg.data['data_loader'])
    attr_loader_cfg.update({'shuffle': False})
    attr_loader = DataLoader(attr_set, **attr_loader_cfg)

    attibuter = build_attributor(
        cfg.attributor, default_args=dict(device=device))
    attibuter.estimate(est_loader, cfg.estimation_cfg)
    evaluator = SanityCheck(attibuter)

    results = {}
    try:
        for batch in tqdm(attr_loader, total=len(attr_loader)):
            inputs = batch['input']
            targets = batch['target']
            input_names = batch['input_name']

            for input_tensor, target, input_name in zip(
                    inputs, targets, input_names):
                input_tensor = input_tensor.to(device)
                target = target.item()

                heatmap_path = osp.join(heatmap_dir, input_name + '.png')
                assert osp.exists(heatmap_path)
                heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)

                ssim_dict = evaluator.evaluate(
                    heatmap=heatmap,
                    input_tensor=input_tensor,
                    target=target,
                    attribution_cfg=cfg.attribution_cfg,
                    perturb_layers=cfg.sanity_check['perturb_layers'],
                    check=cfg.sanity_check['check'],
                    save_dir=osp.join(work_dir, input_name),
                    save_heatmaps=save_heatmaps)
                results.update({input_name: ssim_dict['ssim_all']})
                gc.collect()
    except (KeyboardInterrupt, AssertionError) as e:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return

    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)
    sanity_check(
        cfg=cfg,
        heatmap_dir=args.heatmap_dir,
        work_dir=args.work_dir,
        file_name=args.file_name,
        scores_file=args.scores_file,
        scores_threshold=args.scores_threshold,
        num_samples=args.num_samples,
        save_heatmaps=args.save_heatmaps,
        device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
