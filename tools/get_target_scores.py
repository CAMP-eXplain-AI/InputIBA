from torch.utils.data import DataLoader
import os
import os.path as osp
from argparse import ArgumentParser
import mmcv
from iba.models import build_classifiers
from tqdm import tqdm
from iba.datasets import build_dataset
import torch


def parse_args():
    parser = ArgumentParser(
        'Get score of target class for each sample in the validation set')
    parser.add_argument('config', help='configuration file')
    parser.add_argument('work_dir', help='working directory')
    parser.add_argument(
        'file_name',
        help='file name (e.g. xxx.json) for the result file to be saved')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    args = parser.parse_args()
    return args


def get_target_scores(cfg, work_dir, file_name, device='cuda:0'):
    assert cfg.attributor.get('use_softmax', True), "Currently only support multi-class classification settings," \
                                                    "so use_softmax must be True."
    mmcv.mkdir_or_exist(work_dir)
    val_set = build_dataset(cfg.data['val'])
    val_loader = DataLoader(val_set, **cfg.data['data_loader'])

    classifer = build_classifiers(cfg.attributor['classifier']).to(device)
    classifer.eval()

    res = {}
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            input_names = batch['input_name']

            preds = classifer(inputs)

            preds = torch.softmax(preds, -1)
            target_preds = preds[torch.arange(targets.shape[0]),
                                 targets].detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            res_dict = {
                name: {
                    'target': int(t),
                    'pred': float(p)
                } for name, t, p in zip(input_names, targets, target_preds)
            }
            res.update(res_dict)

    mmcv.dump(res, osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    get_target_scores(cfg=cfg,
                      work_dir=args.work_dir,
                      file_name=args.file_name,
                      device=f"cuda:{args.gpu_id}")


if __name__ == '__main__':
    main()
