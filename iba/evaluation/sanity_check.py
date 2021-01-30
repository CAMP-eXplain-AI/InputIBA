from ..models.net import Net
import torch


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
        # TODO initialize for sequential
        if isinstance(layer, torch.nn.Sequential):
            layer.initialize()
        torch.nn.init.normal(layer.weight, mean=0, std=1)


def sanity_check(net=None, net_kwparams={}, positions=None, check_image_ib=False, check_gan=False):
    """
    random perturb weights before go through a certain submodule, then evaluate generated heatmap
    Returns:
    """
    if net is None:
        perturb_net = Net(net_kwparams)
    else:
        # only copy used model instead of altering directly
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