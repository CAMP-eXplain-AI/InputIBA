import torch
import torch.nn as nn


class BaseWelfordEstimator(nn.Module):
    """
    Estimates the mean and standard derivation.
    For the algorithm see
    ``https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance``.

    Example:
        Given a batch of images ``imgs`` with shape ``(10, 3, 64, 64)``,
        the mean and std could be estimated as follows::

            # exemplary data source: 5 batches of size 10,
            filled with random data

            batch_generator = (torch.randn(10, 3, 64, 64) for _ in range(5))

            estim = WelfordEstimator(3, 64, 64)
            for batch in batch_generator:
                estim(batch)

            # returns the estimated mean
            estim.mean()

            # returns the estimated std
            estim.std()

            # returns the number of samples, here 10
            estim.n_samples()

            # returns a mask with active neurons
            estim.active_neurons()
    """

    def __init__(self):
        super().__init__()
        self.device = None  # Defined on first forward pass
        self.shape = None  # Defined on first forward pass
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))

    def _init(self, shape, device):
        self.device = device
        self.shape = shape
        self.register_buffer('m', torch.zeros(*shape))
        self.register_buffer('s', torch.zeros(*shape))
        self.register_buffer('_neuron_nonzero',
                             torch.zeros(*shape, dtype=torch.long))
        self.to(device)

    def forward(self, x):
        """ Update estimates without altering x """
        pass

    def n_samples(self):
        """ Returns the number of seen samples. """
        return int(self._n_samples.item())

    def mean(self):
        """ Returns the estimate of the mean. """
        return self.m

    def std(self):
        """returns the estimate of the standard derivation."""
        return torch.sqrt(self.s / (self._n_samples.float() - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.float() /
                self._n_samples.float()) > threshold
