import torch

from IBA import pytorch
from IBA.gan import WGAN_CP
from IBA.pytorch_img_iba import Image_IBA

class Net:
    def __init__(self, image=None, target=None, model=None, IBA=None, model_loss_closure=None,
                 image_ib_beta=None, image_ib_optimization_step=None, image_ib_reverse_mask=None, dev=None):
        """
        initialize net by create essential parameter
        """
        # information bottleneck
        self.image = image.to(self.dev)
        assert target is not None, "Please give a target label"
        self.target = target
        self.model = model
        self.IBA = IBA
        self.dev = dev
        if model_loss_closure is None:
            # use default loss if not given (softmax cross entropy)
            self.model_loss_closure = lambda x: -torch.log_softmax(self.model(x), 1)[:, target].mean()

        # GAN
        self.gan = None

        # image level information bottleneck
        self.image_ib_beta = image_ib_beta
        self.image_ib_optimization_step = image_ib_optimization_step
        self.image_ib_reverse_mask = image_ib_reverse_mask

        # results
        self.ib_heatmap = None
        self.image_ib_heatmap = None

    def train(self):
        pass

    def train_ib(self):
        self.ib_heatmap = self.IBA.analyze(self.image[None], self.model_loss_closure)

    def train_gan(self):
        # initialize GAN every time before training
        self.gan = WGAN_CP(self.model, "features[17]",
                image=self.image, feature_mask=self.IBA.capacity(), generator_iters=50,
                feature_noise_mean=self.IBA.estimator.mean(), feature_noise_std=self.IBA.estimator.std(), dev=self.dev)

        # train
        self.gan.train(self.dev)

    def train_image_ib(self):
        # get learned parameters from GAN
        image_mask = self.gan.G.image_mask().clone().detach()
        img_noise_std = self.gan.G.eps
        img_moise_mean = self.gan.G.mean

        # initialize image IBA
        img_iba = Image_IBA(image=self.image.to(self.dev),
                            image_mask=image_mask,
                            img_eps_std=img_noise_std,
                            img_eps_mean=img_moise_mean,
                            beta=self.image_ib_beta,
                            optimization_steps=self.image_ib_optimization_step,
                            reverse_lambda=self.image_ib_reverse_mask).to(self.dev)

        # train image ib
        self.image_ib_heatmap = img_iba.analyze(self.image[None], self.model_loss_closure)

    def plot_feature_mask(self):
        pass