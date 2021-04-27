from captum.attr import GuidedBackprop
from captum.attr import DeepLiftShap
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.extremal_perturbation import extremal_perturbation
from torch.utils.data import DataLoader
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
    parser.add_argument('work_dir', help='working directory')
    parser.add_argument('method', type=str, help='baseline method')
    parser.add_argument('--saliency-layer',
                        type=str,
                        default='features.30',
                        help='Saliency layer of Grad-Cam, only useful when method is grad_cam')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU Id')
    args = parser.parse_args()
    return args


class Baseline:
    method_pool = ['ex_perturb',
                   'grad_cam',
                   'deep_shap',
                   'guided_bp']

    def __init__(self, classifier, method, saliency_layer=None):
        assert method in self.method_pool, f"Invalid method: {method}"
        self.classifier = classifier
        self.method = method
        if method == "ex_perturb":
            def make_attribution(classifier):
                def attribution_func(input, target):
                    saliency_map, _ = extremal_perturbation(classifier, input, target)
                    return saliency_map
                return attribution_func
            self.attribute = make_attribution(self.classifier)

        elif method == "grad_cam":
            assert saliency_layer,  "Please give a saliency layer!"
            def make_attribution(classifier, saliency_layer):
                def attribution_func(input, target):
                    saliency_map = grad_cam(classifier, input, target, saliency_layer=saliency_layer)
                    return saliency_map
                return attribution_func
            self.attribute = make_attribution(self.classifier, saliency_layer)

        elif method == "deep_shap":
            self.attribute = DeepLiftShap(self.classifier).attribute

        elif method == "guided_bp":
            self.attribute = GuidedBackprop(self.classifier).attribute

        else:
            raise ValueError(f'Invalid method: {method}')

    def make_attribution(self, input_tensor, target, **kwargs):
        return self.attribute(input_tensor, target=target, **kwargs)


def train_baseline(cfg, work_dir, method, saliency_layer=None, device='cuda:0'):
    val_set = build_dataset(cfg.data['val'])
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    classifier = build_classifiers(cfg.attributor['classifier']).to(device)
    classifier.eval()
    baseline = Baseline(classifier, method, saliency_layer)

    for batch in tqdm(val_loader, total=len(val_loader)):
        inputs = batch['input']
        targets = batch['target']
        input_names = batch['input_name']
        for input_tensor, target, input_name in zip(inputs, targets, input_names):
            input_tensor = input_tensor.to(device).unsqueeze(0)
            target = target.item()
            if method == 'deep_shap':
                base_distribution = input_tensor.new_zeros((10, ) + input_tensor.shape[1:])
                attr_map = baseline.make_attribution(input_tensor,
                                                     target,
                                                     baselines=base_distribution)
                attr_map = attr_map.detach().cpu().numpy()
            else:
                attr_map = baseline.make_attribution(input_tensor, target).detach().cpu().numpy()
            attr_map = attr_map.mean((0, 1))
            attr_map = attr_map / attr_map.max()

            out_file = osp.join(work_dir, input_name)
            VisionAttributor.show_mask(attr_map, show=False, out_file=out_file)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    mmcv.mkdir_or_exist(args.work_dir)
    train_baseline(cfg=cfg,
                   work_dir=args.work_dir,
                   method=args.method,
                   saliency_layer=args.saliency_layer,
                   device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()