import torch

from .pytorch import tensor_to_np_img
from .gan import WGAN_CP
from .pytorch_img_iba import Image_IBA
import matplotlib.pyplot as plt


class Net:
    # TODO rename class
    def __init__(self,
                 image=None,
                 target=None,
                 model=None,
                 IBA=None,
                 model_loss_closure=None,
                 generator_iters=10,
                 image_ib_beta=10,
                 image_ib_optimization_step=40,
                 image_ib_reverse_mask=False,
                 dev=None):
        """
        initialize net by create essential parameters
        """
        # general setting
        self.dev = dev
        self.image = image.to(self.dev)
        assert target is not None, "Please give a target label"
        self.target = target
        self.model = model

        # information bottleneck
        self.IBA = IBA
        if model_loss_closure is None:
            # use default loss if not given (softmax cross entropy)
            self.model_loss_closure = lambda x: -torch.log_softmax(
                self.model(x), 1)[:, target].mean()

        # GAN
        self.generator_iters = generator_iters
        self.gan = None

        # image level information bottleneck
        self.image_ib_beta = image_ib_beta
        self.image_ib_optimization_step = image_ib_optimization_step
        self.image_ib_reverse_mask = image_ib_reverse_mask
        self.image_ib = None

        # results
        self.ib_heatmap = None
        self.image_ib_heatmap = None

    def train(self):
        print("\nTraining on IB\n")
        self.train_ib()
        print("\nTraining on GAN\n")
        self.train_gan()
        print("\nTraining on image IB\n")
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
                           "features[17]",
                           image=self.image,
                           feature_mask=self.IBA.capacity(),
                           generator_iters=self.generator_iters,
                           feature_noise_mean=self.IBA.estimator.mean(),
                           feature_noise_std=self.IBA.estimator.std(),
                           dev=self.dev)

        # train
        self.gan.train(self.dev)

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
            image=self.image.to(self.dev),
            image_mask=image_mask,
            img_eps_std=img_noise_std,
            img_eps_mean=img_moise_mean,
            beta=self.image_ib_beta,
            optimization_steps=self.image_ib_optimization_step,
            reverse_lambda=self.image_ib_reverse_mask).to(self.dev)

        # train image ib
        self.image_ib_heatmap = self.image_ib.analyze(self.image[None],
                                                      self.model_loss_closure)

    def plot_image(self, label=None):
        """
        plot the image for interpretation
        Returns:
        """
        np_img = tensor_to_np_img(self.net.image)
        if label is not None:
            plt.title(label)
        else:
            plt.title("class {}".format(self.target))
        plt.imshow(np_img)

    def plot_feature_mask(self, upscale=False):
        """
        Plot feature mask from IB either in original size or upscaled to image size
        Returns: None
        """
        if not upscale:
            img_tensor_mean = self.IBA.capacity().sum(0)
            plt.figure()
            plt.imshow(img_tensor_mean.clone().detach().cpu().numpy())
            plt.title('Feature Map Mask')
        else:
            plt.figure()
            plt.imshow(self.ib_heatmap)
            plt.title('Upscaled Feature Map Mask')

    def plot_image_mask(self):
        """
        Plot image mask learned from image IB
        Returns: None
        """
        plt.figure()
        img_tensor = self.image_ib.sigmoid(
            self.image_ib.alpha).detach().cpu().numpy().mean(0).mean(0)
        plt.imshow(img_tensor)

    def plot_generated_image_mask(self):
        """
        Plot generated image mask learned from GAN
        Returns: None
        """
        img_tensor = self.gan.G.image_mask().clone().detach().cpu().numpy()
        img_tensor = img_tensor.mean(0).mean(0)
        plt.imshow(img_tensor)

    def save_mask_as_img(self):
        ## TODO save learned mask with proper name in some folder
        pass

    def plot_generated_image_mask_history(self):
        ## TODO plot generated image mask during learning (for debug and for evaluation)
        pass
