import torch

from .pytorch import tensor_to_np_img
from .gan import WGAN_CP
from .pytorch_img_iba import Image_IBA
import matplotlib.pyplot as plt
from ..utils import get_logger
import os.path as osp
import mmcv
import numpy as np
from PIL import Image


class Net:
    # TODO rename class
    # TODO remove image and device from __init__ to separate them from hyper parameters
    def __init__(self,
                 image=None,
                 target=None,
                 model=None,
                 position=None,
                 IBA=None,
                 model_loss_closure=None,
                 epochs=10,
                 image_ib_beta=10,
                 image_ib_opt_steps=40,
                 image_ib_reverse_mask=False,
                 device=None):
        """
        initialize net by create essential parameters
        """
        # general setting
        self.device = device
        self.image = image.to(self.device)
        assert target is not None, "Please give a target label"
        self.target = target
        self.model = model
        # TODO rename position to align with the argument in __init__ of WGAN_CP
        self.position = position

        # information bottleneck
        self.IBA = IBA
        if model_loss_closure is None:
            # use default loss if not given (softmax cross entropy)
            self.model_loss_closure = lambda x: -torch.log_softmax(
                self.model(x), 1)[:, target].mean()

        # GAN
        self.gan_epochs = epochs
        self.gan = None

        # image level information bottleneck
        self.image_ib_beta = image_ib_beta
        self.image_ib_opt_steps = image_ib_opt_steps
        self.image_ib_reverse_mask = image_ib_reverse_mask
        self.image_ib = None

        # results
        self.ib_heatmap = None
        self.image_ib_heatmap = None

    def train(self, logger=None):
        if logger is None:
            logger = get_logger('iba')
        logger.info("Training on IB")
        self.train_ib()
        logger.info("Training on GAN")
        self.train_gan()
        logger.info("Training on image IB")
        self.train_image_ib()

    def train_ib(self):
        self.ib_heatmap = self.IBA.analyze(self.image[None],
                                           self.model_loss_closure)

    def train_gan(self):
        """
        Train a GAN to get generated image mask
        Returns: None
        """
        # initialize GAN every time before training
        self.gan = WGAN_CP(self.model,
                           self.position,
                           image=self.image,
                           feature_mask=self.IBA.capacity(),
                           epochs=self.gan_epochs,
                           feature_noise_mean=self.IBA.estimator.mean(),
                           feature_noise_std=self.IBA.estimator.std(),
                           device=self.device)

        # train
        self.gan.train(self.device)

    def train_image_ib(self):
        """
        Train image information bottleneck based on result from GAN
        Returns: None
        """
        # get learned parameters from GAN
        image_mask = self.gan.G.image_mask().clone().detach()
        img_noise_std = self.gan.G.eps
        img_moise_mean = self.gan.G.mean

        # initialize image iba
        self.image_ib = Image_IBA(
            image=self.image.to(self.device),
            image_mask=image_mask,
            img_eps_std=img_noise_std,
            img_eps_mean=img_moise_mean,
            beta=self.image_ib_beta,
            optimization_steps=self.image_ib_opt_steps,
            reverse_lambda=self.image_ib_reverse_mask).to(self.device)

        # train image ib
        self.image_ib_heatmap = self.image_ib.analyze(self.image[None],
                                                      self.model_loss_closure)

    @property
    def attribution_map(self):
        """
        get image level attribution map
        """
        return self.image_ib.sigmoid(self.image_ib.alpha).squeeze()

    def plot_image(self, label=None):
        """
        plot the image for interpretation
        """
        np_img = tensor_to_np_img(self.image)
        if label is not None:
            plt.title(label)
        else:
            plt.title("class {}".format(self.target))
        plt.axis('off')
        plt.imshow(np_img)

    def show_feat_mask(self, upscale=False, show=False, out_file=None):
        if not upscale:
            mask = self.IBA.capacity().sum(0).clone().detach().cpu().numpy()
        else:
            mask = self.ib_heatmap
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        self._show_mask(mask, show=show, out_file=out_file)

    def show_gen_img_mask(self, show=False, out_file=None):
        mask = self.gan.G.image_mask().clone().detach().cpu().mean([0, 1]).numpy()
        self._show_mask(mask, show=show, out_file=out_file)

    def show_img_mask(self, show=False, out_file=None):
        mask = self.image_ib.sigmoid(
            self.image_ib.alpha).detach().cpu().mean([0, 1]).numpy()
        self._show_mask(mask, show=show, out_file=out_file)

    @staticmethod
    def _show_mask(mask, show=False, out_file=None):
        plt.imshow(mask)
        plt.axis('off')

        if out_file is not None:
            dir_name = osp.abspath(osp.dirname(out_file))
            mmcv.mkdir_or_exist(dir_name)
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask, mode='L')
            mask.save(out_file + '.png')
            plt.savefig(out_file + '.JPEG', bbox_inches='tight', pad_inches=0)
            if not show:
                plt.close()
        if show:
            plt.show()