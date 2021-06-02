import os
import torch
from torch import nn
from ..model_zoo import MODELS


@MODELS.register_module()
class DeepLSTM(nn.Module):
    def __init__(self, vocab_size=25002, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=4,
                 bidirectional=False, dropout=0.5, pad_idx=1, pretrained=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn_1 = nn.LSTM(embedding_dim,
                             hidden_dim,
                             num_layers=n_layers,
                             bidirectional=bidirectional,
                             dropout=dropout)
        self.rnn_2 = nn.LSTM(hidden_dim,
                             hidden_dim,
                             num_layers=n_layers,
                             bidirectional=bidirectional,
                             dropout=dropout)
        self.rnn_3 = nn.LSTM(hidden_dim,
                             hidden_dim,
                             num_layers=n_layers,
                             bidirectional=bidirectional,
                             dropout=dropout)
        self.rnn_4 = nn.LSTM(hidden_dim,
                             hidden_dim,
                             num_layers=n_layers,
                             bidirectional=bidirectional,
                             dropout=dropout)

        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights(pretrained)

    def _initialize_weights(self, pretrained=None):
        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)
        else:
            #TODO add Error or init randomly
            pass

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, _ = self.rnn_1(packed_embedded)
        packed_output, _ = self.rnn_2(packed_output)
        packed_output, _ = self.rnn_3(packed_output)
        packed_output, (hidden, cell) = self.rnn_4(packed_output)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)

    # returns the word embedding given text
    # this function is only needed for evaluation of attribution method
    def forward_embedding_only(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        return embedded

    # returns logit given word embedding
    # this function is only needed for evaluation of attribution method
    def forward_no_embedding(self, embedding, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(embedding)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, _ = self.rnn_1(packed_embedded)
        packed_output, _ = self.rnn_2(packed_output)
        packed_output, _ = self.rnn_3(packed_output)
        packed_output, (hidden, cell) = self.rnn_4(packed_output)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)