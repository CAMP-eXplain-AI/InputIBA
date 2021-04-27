import torch.nn as nn
from .base_estimator import BaseWelfordEstimator


class VisionWelfordEstimator(BaseWelfordEstimator):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Update estimates without altering x """
        if self.shape is None:
            # Initialize runnnig mean and std on first datapoint
            self._init(x.shape[1:], x.device)
        for xi in x:
            self._neuron_nonzero += (xi != 0.).long()
            old_m = self.m.clone()
            self.m = self.m + (xi - self.m) / (self._n_samples.float() + 1)
            self.s = self.s + (xi - self.m) * (xi - old_m)
            self._n_samples += 1
        return x


class NLPWelfordEstimator(BaseWelfordEstimator):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Update estimates without altering x """
        # only take packed output and unpack for further usage
        output_padded, text_lengths = nn.utils.rnn.pad_packed_sequence(x[0])
        output_padded = output_padded.reshape(-1, output_padded.shape[-1])
        if self.shape is None:
            # Initialize runnnig mean and std on first datapoint
            self._init(output_padded.shape[1:], output_padded.device)
        for xi in output_padded:
            self._neuron_nonzero += (xi != 0.).long()
            old_m = self.m.clone()
            self.m = self.m + (xi - self.m) / (self._n_samples.float() + 1)
            self.s = self.s + (xi - self.m) * (xi - old_m)

        # consider a sentence as one sample for estimation
        self._n_samples += len(text_lengths)
        # x = nn.utils.rnn.pack_padded_sequence(x, text_lengths.to('cpu'))
        return x
