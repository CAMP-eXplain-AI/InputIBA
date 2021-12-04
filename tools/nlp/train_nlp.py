import os
import os.path as osp
import warnings

import gc
import mmcv
import torch
from argparse import ArgumentParser
from mmcv.runner.utils import set_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from input_iba.datasets import build_dataset, nlp_collate_fn
from input_iba.models import build_attributor


def parse_args():
    parser = ArgumentParser('Train the attributor and get the attribution '
                            'maps')
    parser.add_argument('config', help='configuration file')
    parser.add_argument(
        '--work-dir',
        help='directory to save output files',
        default=os.getcwd())
    parser.add_argument('--gpu-id', help='gpu id', type=int, default=0)
    parser.add_argument(
        '--pbar',
        action='store_true',
        help='Whether to use a progressbar to track the main loop')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')

    args = parser.parse_args()
    return args


def train(config, work_dir, gpu_id=0, pbar=False):
    cfg = mmcv.Config.fromfile(config)
    # the batch_size of input_iba and feat_iba must be the same
    input_iba_batch_size = cfg.attribution_cfg['input_iba']['batch_size']
    feat_iba_batch_size = cfg.attribution_cfg['feat_iba']['batch_size']
    assert input_iba_batch_size == feat_iba_batch_size, \
        "batch_size of input_iba and feat_iba must be the same, " \
        f"but got input_iba: {input_iba_batch_size}, " \
        f"feat_iba:{feat_iba_batch_size}"

    mmcv.mkdir_or_exist(work_dir)
    if len(os.listdir(work_dir)) > 0:
        warnings.warn('The working directory is not empty')
    mmcv.mkdir_or_exist(osp.join(work_dir, 'input_masks'))
    mmcv.mkdir_or_exist(osp.join(work_dir, 'feat_masks'))

    cfg.dump(osp.join(work_dir, 'config.py'))
    logger = mmcv.get_logger(
        'iba', log_file=osp.join(work_dir, 'log_file.log'))
    device = f'cuda:{gpu_id}'
    train_attributor(cfg, logger, work_dir=work_dir, device=device, pbar=pbar)


def train_attributor(cfg, logger, work_dir, device='cuda:0', pbar=False):
    est_set = build_dataset(cfg.data['estimation'])
    attr_set = build_dataset(cfg.data['attribution'])

    # Only use data loader at estimation stage. At attribution stage,
    # samples will be loaded separately. This aims to avoid the padding
    # operation at the attribution stage
    est_loader = DataLoader(
        est_set, collate_fn=nlp_collate_fn, **cfg.data['data_loader'])

    attributor = build_attributor(
        cfg.attributor, default_args=dict(device=device))
    attributor.estimate(est_loader, cfg.estimation_cfg)

    if pbar:
        bar = tqdm()
    else:
        bar = None

    for sample in attr_set:
        input_tensor = sample['input']
        input_text = sample['input_text']
        input_name = sample['input_name']
        target = sample['target']
        if target > 1:
            raise RuntimeError('Currently only binary classification is '
                               'supported for NLP tasks. But found target '
                               'larger than 1')
        feat_iba_batch_size = cfg.attribution_cfg['feat_iba']['batch_size']
        target = torch.tensor([target]).expand((feat_iba_batch_size, -1))
        target = target.to(torch.float32).to(device)

        input_tensor = input_tensor.to(device)
        attributor.set_text(input_text)

        feat_mask_file = osp.join(work_dir, 'feat_masks', input_name)
        input_mask_file = osp.join(work_dir, 'input_masks', input_name)

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
