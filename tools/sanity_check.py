from torch.utils.data import DataLoader
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from iba.models import Attributer
from iba.datasets import build_dataset
from iba.evaluation import SanityCheck
import cv2
import gc
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser('Sanity check')
    parser.add_argument('config',
                        help='config file of the attribution method')
    parser.add_argument('heatmap_dir',
                        help='directory of the heatmaps')
    parser.add_argument('work_dir',
                        help='directory to save the result file')
    parser.add_argument('file_name',
                        help='file name fo saving the results')
    parser.add_argument('--save-heatmaps',
                        action='store_true',
                        default=False,
                        help='Whether to save the heatmaps produced by the perturbed models')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU id')
    args = parser.parse_args()
    return args


def sanity_check(cfg,
                 heatmap_dir,
                 work_dir,
                 file_name,
                 save_heatmaps=False,
                 device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)
    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    train_loader = DataLoader(train_set, **cfg.data['data_loader'])
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    attibuter = Attributer(cfg.attributer, device=device)
    attibuter.estimate(train_loader, cfg.estimation_cfg)
    evaluator = SanityCheck(attibuter)

    results = {}
    try:
        for batch in tqdm(val_loader, total=len(val_loader)):
            imgs = batch['img']
            targets = batch['target']
            img_names = batch['img_name']

            for img, target, img_name in zip(imgs, targets, img_names):
                img = img.to(device)
                target = target.item()
                heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'), cv2.IMREAD_UNCHANGED)

                ssim_dict = evaluator.evaluate(heatmap=heatmap,
                                               img=img,
                                               target=target,
                                               attribution_cfg=cfg.attribution_cfg,
                                               perturb_layers=cfg.sanity_check['perturb_layers'],
                                               check=cfg.sanity_check['check'],
                                               save_dir=osp.join(work_dir, img_name),
                                               save_heatmaps=save_heatmaps)
                results.update({img_name: ssim_dict['ssim_all']})
                gc.collect()
    except KeyboardInterrupt:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return

    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    sanity_check(cfg=cfg,
                 heatmap_dir=args.heatmap_dir,
                 work_dir=args.work_dir,
                 file_name=args.file_name,
                 save_heatmaps=args.save_heatmaps,
                 device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()