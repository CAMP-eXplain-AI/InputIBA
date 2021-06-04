from torch.utils.data import DataLoader, Subset
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from mmcv.runner.utils import set_random_seed
from iba.models import build_attributor
from tqdm import tqdm
from iba.datasets import build_dataset
import torch
import numpy as np
import gc


def parse_args():
    parser = ArgumentParser('train the attributior and get the attribution '
                            'maps')
    parser.add_argument('config', help='configuration file')
    parser.add_argument(
        '--work-dir',
        help='directory to save the output files',
        default=os.getcwd())
    parser.add_argument('--gpu-id', help='gpu id', type=int, default=0)
    parser.add_argument(
        '--out-style',
        help='Structure of output folders that store the attribution maps',
        choices=['image_folder', 'single_folder'],
        default='single_folder')
    parser.add_argument(
        '--pbar',
        action='store_true',
        help='Whether to use a progressbar to track the main loop')
    parser.add_argument(
        '--subset-file',
        help='A txt file, where each line stores the sample index in subset. '
        'Attribution is only applied on this subset')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')

    args = parser.parse_args()
    return args


def train(config,
          work_dir,
          gpu_id=0,
          out_style='single_folder',
          pbar=False,
          subset_file=None):
    cfg = mmcv.Config.fromfile(config)
    mmcv.mkdir_or_exist(work_dir)
    if len(os.listdir(work_dir)) > 0:
        warnings.warn('The working directory is not empty!')
    mmcv.mkdir_or_exist(osp.join(work_dir, 'input_masks'))
    mmcv.mkdir_or_exist(osp.join(work_dir, 'feat_masks'))

    cfg.dump(osp.join(work_dir, 'config.py'))
    logger = mmcv.get_logger(
        'iba', log_file=osp.join(work_dir, 'log_file.log'))
    device = f'cuda:{gpu_id}'
    train_attributor(
        cfg,
        logger,
        work_dir=work_dir,
        device=device,
        out_style=out_style,
        pbar=pbar,
        subset_file=subset_file)


def train_attributor(cfg: mmcv.Config,
                     logger,
                     work_dir,
                     device='cuda:0',
                     out_style='single_folder',
                     pbar=False,
                     subset_file=None):
    assert out_style in ('single_folder', 'image_folder'), \
        f"Invalid out_style, should be one of " \
        f"('single_folder', 'image_folder'), but got {out_style}"
    est_set = build_dataset(cfg.data['estimation'])
    attr_set = build_dataset(cfg.data['attribution'])
    if subset_file is not None:
        subset_inds = np.loadtxt(subset_file, dtype=int)
        attr_set = Subset(attr_set, indices=subset_inds)
    est_loader = DataLoader(est_set, **cfg.data['data_loader'])
    attr_loader_cfg = deepcopy(cfg.data['data_loader'])
    attr_loader_cfg.update({'shuffle': False})
    attr_loader = DataLoader(attr_set, **attr_loader_cfg)

    attributor = build_attributor(
        cfg.attributor, default_args=dict(device=device))
    attributor.estimate(est_loader, cfg.estimation_cfg)

    if pbar:
        bar = tqdm(attr_loader, total=len(attr_loader))
    else:
        bar = None

    for batch in attr_loader:
        inputs = batch['input']
        targets = batch['target']
        input_names = batch['input_name']
        for input_tensor, target, input_name in zip(inputs, targets,
                                                    input_names):
            input_tensor = input_tensor.to(device)
            if target.nelement() == 1:
                # multi-class classification,
                # target of one sample is an integer
                target = target.item()
            else:
                # multi-label classification,
                # target of one sample is an one-hot vector
                target = target.to(device)

            if out_style == 'single_folder':
                feat_mask_file = osp.join(work_dir, 'feat_masks', input_name)
                input_mask_file = osp.join(work_dir, 'input_masks', input_name)
            else:
                if isinstance(target, torch.Tensor) and target.nelement() > 1:
                    raise RuntimeError('For multi-label classification, '
                                       'saving the attribution maps with '
                                       'image folder style is not possible')
                if isinstance(attr_set, Subset):
                    sub_dir = attr_set.dataset.ind_to_cls[target]  # noqa
                else:
                    sub_dir = attr_set.ind_to_cls[target]

                img_mask_dir = osp.join(work_dir, 'input_masks', sub_dir)
                feat_mask_dir = osp.join(work_dir, 'feat_masks', sub_dir)
                mmcv.mkdir_or_exist(img_mask_dir)
                mmcv.mkdir_or_exist(feat_mask_dir)

                feat_mask_file = osp.join(feat_mask_dir, input_name)
                input_mask_file = osp.join(img_mask_dir, input_name)

            attributor.make_attribution(
                input_tensor,
                target,
                attribution_cfg=cfg.attribution_cfg,
                logger=logger)
            attributor.show_feat_mask(
                out_file=feat_mask_file,
                **cfg.attribution_cfg.get('feat_mask', {}))
            attributor.show_input_mask(
                out_file=input_mask_file,
                **cfg.attribution_cfg.get('input_mask', {}))
            gc.collect()

            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()


def main():
    args = parse_args()
    args = vars(args)
    set_random_seed(seed=args.pop('seed'))
    train(**args)


if __name__ == '__main__':
    main()
