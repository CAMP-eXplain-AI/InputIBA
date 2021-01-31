from iba.models.net import Net
import torch
from torch.nn import init
from copy import deepcopy


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(m.bias)


def perturb_model(model, positions=()):
    for position in positions:
        if "[" in position:
            # read the index
            layer_name = position.split("[")[0]
            idx_feature = int(position.split("[")[1].split("]")[0])
            layer = getattr(model, str(layer_name))[idx_feature]
        else:
            layer = getattr(model, position)

        # initialize weights
        layer.apply(weights_init)


def sanity_check(net=None, net_kwparams={}, positions=None, check_image_ib=False, check_gan=False):
    """
    random perturb weights before go through a certain submodule, then evaluate generated heatmap
    Returns:
    """
    if net is None:
        perturb_net = Net(net_kwparams)
    else:
        perturb_net = net

    # training with perturbed model on required stage
    perturb_net.train_ib()
    if check_gan:
        perturb_model(perturb_net.model, positions)
    perturb_net.train_gan()
    if check_image_ib:
        perturb_model(perturb_net.model, positions)
    perturb_net.train_image_ib()

    perturb_net.plot_image_mask()