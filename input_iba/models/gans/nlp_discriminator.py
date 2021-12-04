import torch.nn as nn

from .builder import DISCRIMINATORS


@DISCRIMINATORS.register_module()
class NLPDiscriminator(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        # Input_dim = channels (SentenceLengthxHiddenSize)
        # Output_dim = 1
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=1)

        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.output = nn.Sequential(
            # The output of discriminator is no longer a probability,
            # we do not apply sigmoid at the output of discriminator.
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        output, (hidden, cell) = self.rnn(x)
        return self.output(hidden[-1, :, :])
