from torch.utils.data import DataLoader
import torch
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from mmcv.runner.utils import set_random_seed
from iba.models import build_classifiers
from iba.datasets import build_dataset
from iba.evaluation import InsertionDeletion
from iba.utils import get_valid_set
import cv2
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser('Insertion Deletion evaluation')
    parser.add_argument('config',
                        help='config file of the attribution method')
    parser.add_argument('heatmap_dir',
                        help='directory of the heatmaps')
    parser.add_argument('work_dir',
                        help='directory to save the result file')
    parser.add_argument('file_name',
                        help='file name with extension of the results to be saved')
    parser.add_argument('--scores-file',
                        help='File that records the predicted probability of corresponding target class')
    parser.add_argument('--scores-threshold',
                        type=float,
                        default=0.6,
                        help='Threshold for filtering the samples with low predicted target probabilities')
    parser.add_argument('--num-samples',
                        type=int,
                        default=0,
                        help='Number of samples to check, 0 means checking all the (filtered) samples')
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
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help='Random seed')
    args = parser.parse_args()
    return args


def insertion_deletion(cfg,
                       heatmap_dir,
                       work_dir,
                       file_name,
                       scores_file=None,
                       scores_threshold=0.6,
                       num_samples=0,
                       pixel_batch_size=10,
                       sigma=5.0,
                       device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)
    val_set = build_dataset(cfg.data['val'])

    val_set = get_valid_set(val_set,
                            scores_file=scores_file,
                            scores_threshold=scores_threshold,
                            num_samples=num_samples)

    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    classifer = build_classifiers(cfg.attributor['classifier']).to(device)
    evaluator = InsertionDeletion(classifer,
                                  pixel_batch_size=pixel_batch_size,
                                  sigma=sigma)

    results = {}
    try:
        for batch in tqdm(val_loader, total=len(val_loader)):
            inputs = batch['input']
            targets = batch['target']
            input_names = batch['input_name']

            for input_tensor, target, input_name in zip(inputs, targets, input_names):
                input_tensor = input_tensor.to(device)
                target = target.item()
                heatmap = cv2.imread(osp.join(heatmap_dir, input_name + '.png'),
                                     cv2.IMREAD_UNCHANGED)
                heatmap = torch.from_numpy(heatmap).to(input_tensor) / 255.0

                res_single = evaluator.evaluate(heatmap, input_tensor, target)
                ins_auc = res_single['ins_auc']
                del_auc = res_single['del_auc']

                results.update(
                    {input_name: dict(ins_auc=ins_auc, del_auc=del_auc)})
    except KeyboardInterrupt:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return

    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)
    insertion_deletion(cfg=cfg,
                       heatmap_dir=args.heatmap_dir,
                       work_dir=args.work_dir,
                       file_name=args.file_name,
                       scores_file=args.scores_file,
                       scores_threshold=args.scores_threshold,
                       num_samples=args.num_samples,
                       pixel_batch_size=args.pixel_batch_size,
                       sigma=args.sigma,
                       device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
