from torch.utils.data import DataLoader
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from iba.models import build_attributor
from tqdm import tqdm
from iba.datasets import build_dataset
import torch
import gc


def parse_args():
    parser = ArgumentParser('train a model')
    parser.add_argument('config', help='configuration file')
    parser.add_argument('--work-dir',
                        help='working directory',
                        default=os.getcwd())
    parser.add_argument('--gpu-id', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    return args


def train(config, work_dir, gpu_id=0):
    cfg = mmcv.Config.fromfile(config)
    mmcv.mkdir_or_exist(work_dir)
    if len(os.listdir(work_dir)) > 0:
        warnings.warn('The working directory is not empty!')
    cfg.dump(osp.join(work_dir, 'config.py'))

    logger = mmcv.get_logger('iba', log_file=osp.join(work_dir, 'log_file.log'))
    device = f'cuda:{gpu_id}'
    train_attributor(cfg, logger, work_dir=work_dir, device=device)


def train_attributor(cfg: mmcv.Config, logger, work_dir, device='cuda:0'):
    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    train_loader = DataLoader(train_set, **cfg.data['data_loader'])
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    attributor = build_attributor(cfg.attributor, default_args=dict(device=device))
    attributor.estimate(train_loader, cfg.estimation_cfg)

    for batch in tqdm(val_loader, total=len(val_loader)):
        inputs = batch['input']
        targets = batch['target']
        input_names = batch['input_name']
        for input_tensor, target, input_name in zip(inputs, targets, input_names):
            # TODO remove this log message after debug
            logger.info(
                f'allocated memory in MB: '
                f'{int(torch.cuda.memory_allocated(device) / (1024 ** 2))}')
            input_tensor = input_tensor.to(device)
            if target.nelement() == 1:
                target = target.item()
            else:
                target = target.to(device)
            feat_mask_file = osp.join(work_dir, 'feat_masks', input_name)
            input_mask_file = osp.join(work_dir, 'input_masks', input_name)

            attributor.make_attribution(input_tensor,
                                        target,
                                        attribution_cfg=cfg.attribution_cfg,
                                        logger=logger)
            attributor.show_feat_mask(out_file=feat_mask_file,
                                      **cfg.attribution_cfg.get(
                                          'feat_mask', {}))
            attributor.show_img_mask(out_file=input_mask_file,
                                     **cfg.attribution_cfg.get('input_mask', {}))
            gc.collect()


def main():
    args = parse_args()
    train(config=args.config, work_dir=args.work_dir, gpu_id=args.gpu_id)


if __name__ == '__main__':
    main()
