import torch
import mmcv
import cv2
from tqdm import tqdm
from mmcv.runner.utils import set_random_seed
import os.path as osp

from ..iba.models import build_classifiers
from ..iba.datasets import build_dataset
from ..iba.evaluation.nlp.insertion_deletion import NLPInsertionDeletion
from argparse import ArgumentParser

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
    parser.add_argument('--num-samples',
                        type=int,
                        default=2000,
                        help='Number of samples to check, 0 means checking all the (filtered) samples')
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
                       num_samples=0,
                       device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)

    classifier = build_classifiers(cfg.attributor['classifier']).to(device)
    dataset = build_dataset(cfg.data['attribution'])
    insertion_deletion_eval = NLPInsertionDeletion(classifier)

    results = {}
    count = 0
    try:
        for datapoint in tqdm(next(iter(dataset)), total=num_samples):
            count += 1
            input_tensor = datapoint['input'].to(device)
            target = datapoint['target']
            img_name = datapoint['input_name']
            heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'),
                                 cv2.IMREAD_UNCHANGED)
            heatmap = torch.from_numpy(heatmap).to(device) / 255.0

            res_single = insertion_deletion_eval.evaluate(heatmap,
                                                          input_tensor,
                                                          target)
            ins_auc = res_single['ins_auc']
            del_auc = res_single['del_auc']

            results.update(
                {img_name: dict(ins_auc=ins_auc, del_auc=del_auc)})
            if count >= num_samples:
                break
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
                       num_samples=args.num_samples,
                       device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
