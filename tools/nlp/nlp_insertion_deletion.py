import os.path as osp

import cv2
import mmcv
import torch
from argparse import ArgumentParser
from mmcv.runner.utils import set_random_seed
from tqdm import tqdm

from input_iba.datasets import build_dataset
from input_iba.evaluation.nlp.insertion_deletion import NLPInsertionDeletion
from input_iba.models import build_classifiers


def parse_args():
    parser = ArgumentParser('Insertion Deletion evaluation')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument(
        'file_name',
        help='file name with extension of the results to be saved')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2000,
        help='Number of samples to check, 0 means checking all the (filtered) '
        'samples')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    args = parser.parse_args()
    return args


def insertion_deletion(cfg,
                       heatmap_dir,
                       work_dir,
                       file_name,
                       num_samples=0,
                       device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)
    logger = mmcv.get_logger('iba')

    classifier = build_classifiers(cfg.attributor['classifier']).to(device)
    dataset = build_dataset(cfg.data['attribution'])
    insertion_deletion_eval = NLPInsertionDeletion(classifier)

    results = {}
    count = 0
    try:
        for datapoint in tqdm(dataset):
            count += 1
            input_tensor = datapoint['input'].to(device)
            target = datapoint['target']
            input_name = datapoint['input_name']

            heatmap_path = osp.join(heatmap_dir, input_name + '.png')
            assert osp.exists(heatmap_path), \
                f'File {heatmap_path} does not exist or is empty'
            heatmap = cv2.imread(
                osp.join(heatmap_dir, input_name + '.png'),
                cv2.IMREAD_UNCHANGED)
            heatmap = torch.from_numpy(heatmap).to(device) / 255.0

            res_single = insertion_deletion_eval.evaluate(
                heatmap, input_tensor, target)
            ins_auc = res_single['ins_auc']
            del_auc = res_single['del_auc']

            results.update(
                {input_name: dict(ins_auc=ins_auc, del_auc=del_auc)})
            if count >= num_samples:
                break
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
    insertion_deletion(
        cfg=cfg,
        heatmap_dir=args.heatmap_dir,
        work_dir=args.work_dir,
        file_name=args.file_name,
        num_samples=args.num_samples,
        device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
