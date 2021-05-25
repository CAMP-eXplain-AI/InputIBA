from captum.attr import GuidedBackprop
from captum.attr import DeepLiftShap
from captum.attr import IntegratedGradients
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.extremal_perturbation import extremal_perturbation
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import os.path as osp
import mmcv
from tqdm import tqdm
from iba.datasets import build_dataset
from iba.models import build_classifiers, VisionAttributor
from copy import deepcopy
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Train other baselines')
    parser.add_argument('config', help='config file')
    parser.add_argument('method', type=str, help='baseline method')
    parser.add_argument(
        '--work-dir', help='working directory', default=os.getcwd())
    parser.add_argument(
        '--saliency-layer',
        type=str,
        default='features.30',
        help='Saliency layer of Grad-Cam, only useful when method is grad_cam')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU Id')
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
    args = parser.parse_args()
    return args


class Baseline:
    method_pool = [
        'ex_perturb', 'grad_cam', 'deep_shap', 'guided_bp', 'int_grad',
        'random'
    ]

    def __init__(self, classifier, method, saliency_layer=None):
        assert method in self.method_pool, f"Invalid method: {method}"
        self.classifier = classifier
        self.method = method
        if method == "ex_perturb":

            def make_attribution(classifier):

                def attribution_func(input, target):
                    saliency_map, _ = extremal_perturbation(
                        classifier, input, target, areas=[0.3])
                    return saliency_map

                return attribution_func

            self.attribute = make_attribution(self.classifier)

        elif method == "grad_cam":
            assert saliency_layer, "Please give a saliency layer!"

            def make_attribution(classifier, saliency_layer):

                def attribution_func(input, target):
                    saliency_map = grad_cam(
                        classifier,
                        input,
                        target,
                        saliency_layer=saliency_layer)
                    return saliency_map

                return attribution_func

            self.attribute = make_attribution(self.classifier, saliency_layer)

        elif method == "deep_shap":
            self.attribute = DeepLiftShap(self.classifier).attribute

        elif method == "guided_bp":
            self.attribute = GuidedBackprop(self.classifier).attribute

        elif method == "int_grad":
            self.attribute = IntegratedGradients(self.classifier).attribute

        elif method == "random":

            def attribution_func(input, target):
                return torch.randn_like(input) / 2.

            self.attribute = attribution_func

        else:
            raise ValueError(f'Invalid method: {method}')

    def make_attribution(self, input_tensor, target, **kwargs):
        return self.attribute(input_tensor, target=target, **kwargs)


def train_baseline(cfg,
                   method,
                   work_dir,
                   saliency_layer=None,
                   device='cuda:0',
                   out_style='single_folder',
                   pbar=False,
                   subset_file=None):
    assert out_style in ('single_folder', 'image_folder'), \
        f"Invalid out_style, should be one of " \
        f"('single_folder', 'image_folder'), but got {out_style}"
    val_set = build_dataset(cfg.data['val'])
    if subset_file is not None:
        subset_inds = np.loadtxt(subset_file, dtype=int)
        val_set = Subset(val_set, indices=subset_inds)
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    classifier = build_classifiers(cfg.attributor['classifier']).to(device)
    classifier.eval()
    baseline = Baseline(classifier, method, saliency_layer)

    if pbar:
        bar = tqdm(val_loader, total=len(val_loader))
    else:
        bar = None

    for batch in val_loader:
        inputs = batch['input']
        targets = batch['target']
        input_names = batch['input_name']
        for input_tensor, target, input_name in zip(inputs, targets,
                                                    input_names):
            input_tensor = input_tensor.to(device).unsqueeze(0)
            target = target.item()
            if method == 'deep_shap':
                base_distribution = input_tensor.new_zeros(
                    (10, ) + input_tensor.shape[1:])
                attr_map = baseline.make_attribution(
                    input_tensor, target, baselines=base_distribution)
                attr_map = attr_map.detach().cpu().numpy()
            else:
                attr_map = baseline.make_attribution(
                    input_tensor, target).detach().cpu().numpy()
            attr_map = attr_map.mean((0, 1))

            if out_style == 'single_folder':
                out_file = osp.join(work_dir, input_name)
            else:
                if isinstance(val_set, Subset):
                    sub_dir = val_set.dataset.ind_to_cls[target]  # noqa
                else:
                    sub_dir = val_set.ind_to_cls[target]
                mmcv.mkdir_or_exist(osp.join(work_dir, sub_dir))
                out_file = osp.join(work_dir, sub_dir, input_name)
            VisionAttributor.show_mask(attr_map, show=False, out_file=out_file)

            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    mmcv.mkdir_or_exist(args.work_dir)
    train_baseline(
        cfg=cfg,
        method=args.method,
        work_dir=args.work_dir,
        saliency_layer=args.saliency_layer,
        device=f'cuda:{args.gpu_id}',
        out_style=args.out_style,
        pbar=args.pbar,
        subset_file=args.subset_file)


if __name__ == '__main__':
    main()
