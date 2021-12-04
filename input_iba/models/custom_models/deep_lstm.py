import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..model_zoo import MODELS


@MODELS.register_module()
class DeepLSTM(nn.Module):

    def __init__(self,
                 num_classes=2,
                 vocab_size=25002,
                 embedding_dim=100,
                 hidden_dim=256,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.5,
                 pad_idx=1,
                 pretrained=None):
        super().__init__()
        assert num_classes > 1
        if num_classes == 2:
            final_out_dim = 1
        else:
            final_out_dim = num_classes

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn_1 = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout)
        self.rnn_2 = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout)
        self.rnn_3 = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout)
        self.rnn_4 = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, final_out_dim)

        self._initialize_weights(pretrained)

    def _initialize_weights(self, pretrained=None):
        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)
        return self

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        # embedded.shape: (sent_len, batch_size, emb_dim)
        embedded = self.dropout(self.embedding(text))

        # pack sequence, lengths need to be on CPU
        packed_embedded = pack_padded_sequence(
            embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, _ = self.rnn_1(packed_embedded)
        packed_output, _ = self.rnn_2(packed_output)
        packed_output, _ = self.rnn_3(packed_output)
        packed_output, (hidden, cell) = self.rnn_4(packed_output)

        # unpack sequence
        # output.shape: (sent_len, batch_size, hid_dim * num_directions)
        # output over padding tokens are zero tensors
        output, output_lengths = pad_packed_sequence(packed_output)

        # hidden.shape: (num_layers * num_directions, batch_size, hid_dim)
        # cell.shape: (num_layers * num_directions, batch_size, hid_dim)

        # hidden.shape: (batch_size, hid_dim * num_directions)
        hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden)

    def forward_embedding_only(self, text):
        """Returns the word embedding given text this function is only
        needed for evaluation of attribution method.

        Args:
            text (Tensor): input text with shape `(sent_len, batch_size)`.

        Returns:
            Tensor: embeddings.
        """

        embedded = self.embedding(text)

        return embedded

    def forward_no_embedding(self, embedding, text_lengths):
        """Returns logit given word embedding this function is only needed
        for evaluation of attribution method.

        Args:
            embedding (Tensor): embeddings with shape `(sent_len, batch_size,
                emb_dim)`.
            text_lengths (Tensor): text lengths with shape `(batch_size,)`.

        Returns:
            Tensor: predictions with shape `(batch_size, )`.
        """
        # text = [sent len, batch size]

        embedded = self.dropout(embedding)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, _ = self.rnn_1(packed_embedded)
        packed_output, _ = self.rnn_2(packed_output)
        packed_output, _ = self.rnn_3(packed_output)
        packed_output, (hidden, cell) = self.rnn_4(packed_output)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        # output.shape: (sent_len, batch_size, hid_dim * num_directions)
        # output over padding tokens are zero tensors

        # hidden.shape: (num_layers * num_directions, batch_size, hid_dim)
        # cell.shape: (num_layers * num directions, batch_size, hid_dim)

        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)
