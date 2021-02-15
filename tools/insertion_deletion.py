from torch.utils.data import DataLoader, Subset
import torch
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from iba.models import Attributer
from iba.datasets import build_dataset
from iba.evaluation import InsertionDeletion
import cv2
import gc
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation')
    parser.add_argument('config',
                        help='config file of the attribution method')
    parser.add_argument('heatmap_dir',
                        help='directory of the heatmaps')
    parser.add_argument('work_dir',
                        help='directory to save the result file')
    parser.add_argument('file_name',
                        help='file name fo saving the results')
    parser.add_argument('--num-samples',
                        type=int,
                        default=0,
                        help='Number of samples to check, 0 means checking all the samples')
    parser.add_argument('--pixel-batch-size',
                        type=int,
                        default=10,
                        help='Batch size for inserting or deleting the heatmap pixels')
    parser.add_argument('--sigma',
                        type=float,
                        default=5.0,
                        help='Sigma of the gaussian blur filter')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU id')
    args = parser.parse_args()
    return args


def insertion_deletion(cfg,
                       heatmap_dir,
                       work_dir,
                       file_name,
                       num_samples=0,
                       pixel_batch_size=10,
                       sigma=5.0,
                       device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)
    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    if num_samples > 0:
        inds = np.arange(num_samples)
        val_set = Subset(val_set, inds)
    train_loader = DataLoader(train_set, **cfg.data['data_loader'])
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    attributer = Attributer(cfg.attributer, device=device)
    evaluator = InsertionDeletion(attributer,
                                  pixel_batch_size=pixel_batch_size,
                                  sigma=sigma)

    results = {}
    try:
        for batch in tqdm(val_loader, total=(len(val_loader))):
            imgs = batch['img']
            targets = batch['target']
            img_names = batch['img_name']

            for img, target, img_name in zip(imgs, targets, img_names):
                img = img.to(device)
                target = target.item()
                heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'), cv2.IMREAD_UNCHANGED)
                heatmap = torch.from_numpy(heatmap).to(img) / 255.0

                res_single = evaluator.evaluate(heatmap, img, target)
                ins_auc = res_single['ins_auc']
                del_auc = res_single['del_auc']

                results.update({img_name: dict(ins_auc=ins_auc,
                                               del_auc=del_auc)})
    except KeyboardInterrupt:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return

    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    insertion_deletion(cfg=cfg,
                       heatmap_dir=args.heatmap_dir,
                       work_dir=args.work_dir,
                       file_name=args.file_name,
                       num_samples=args.num_samples,
                       pixel_batch_size=args.pixel_batch_size,
                       sigma=args.sigma,
                       device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
