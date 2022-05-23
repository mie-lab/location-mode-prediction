import torch.nn as nn
import numpy as np
import torch
from torch import Tensor

import torch.nn.functional as F


class Deepmove(nn.Module):
    def __init__(self, config) -> None:
        super(Deepmove, self).__init__()
        self.emb_loc = nn.Embedding(config.total_loc_num, config.loc_emb_size)
        self.emb_time = nn.Embedding(60 * 24 // 30, config.time_emb_size)

        # the input size to each layer
        self.d_input = config.loc_emb_size + config.time_emb_size

        self.rnn_encoder = RNN_Classifier(config, self.d_input)
        self.rnn_decoder = RNN_Classifier(config, self.d_input)

        self.attn = Attention()

        self.emb_user = nn.Embedding(config.total_user_num, config.user_emb_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.user_emb_size, config.total_loc_num)

        self.dropout = nn.Dropout(p=0.1)

        # init parameter
        self._init_weights()
        self._init_weights_rnn()

    def forward(self, src, context_dict, device) -> Tensor:

        history, curr = src
        history_context, context_dict = context_dict

        # length of each batch
        history_batch_len = history_context["len"].to(device)
        curr_batch_len = context_dict["len"].to(device)

        # time and user info
        history_time = history_context["time"].to(device)
        curr_time = context_dict["time"].to(device)
        user = context_dict["user"].to(device)

        # embedding
        history_emb = torch.cat([self.emb_loc(history), self.emb_time(history_time)], -1)
        curr_emb = torch.cat([self.emb_loc(curr), self.emb_time(curr_time)], -1)

        # send to rnn
        hidden_history, _ = self.rnn_encoder(history_emb)
        hidden_state, _ = self.rnn_decoder(curr_emb)

        # only take the last timestep as target
        hidden_state = hidden_state.gather(
            0, curr_batch_len.view([1, -1, 1]).expand([1, hidden_state.shape[1], hidden_state.shape[-1]]) - 1
        ).squeeze()

        # get the attention with history hidden state
        _, context = self.attn(hidden_state, hidden_history, hidden_history, history_batch_len, device)
        out = torch.cat((hidden_state, context), 1)

        # concat user embedding
        emb_user = self.emb_user(user)
        out = torch.cat([out, emb_user], -1)
        out = self.dropout(out)

        # with fc output
        return self.fc(out)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def _init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init_weights_rnn(self):
        """Reproduce Keras default initialization weights for consistency with Keras version."""
        ih = (param.data for name, param in self.named_parameters() if "weight_ih" in name)
        hh = (param.data for name, param in self.named_parameters() if "weight_hh" in name)
        b = (param.data for name, param in self.named_parameters() if "bias" in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)


class RNN_Classifier(nn.Module):
    """Baseline LSTM model."""

    def __init__(self, config, d_input):
        super(RNN_Classifier, self).__init__()
        RNNS = ["LSTM", "GRU"]
        self.bidirectional = False
        assert config.rnn_type in RNNS, "Use one of the following: {}".format(str(RNNS))
        rnn_cell = getattr(nn, config.rnn_type)  # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(
            d_input, hidden_size=config.hidden_size, num_layers=1, dropout=0.0, bidirectional=self.bidirectional
        )

    def forward(self, input, hidden=None):
        """Forward pass of the network."""
        return self.rnn(input, hidden)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, keys, values, src_len, device):
        """
        Here we assume q_dim == k_dim (dot product attention).

        Query = [BxQ]
        Keys = [TxBxK]
        Values = [TxBxV]
        Outputs = a:[TxB], lin_comb:[BxV]
        src_len:
           used for masking. NoneType or tensor in shape (B) indicating sequence length
        """
        keys = keys.transpose(0, 1)  # [B*T*H]
        energy = torch.bmm(keys, query.unsqueeze(-1)).transpose(2, 1)  # [B,1,T]

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (keys.size(1) - src_len[b].item()))
            mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(1).to(device)  # [B,1,T]
            energy = energy.masked_fill(mask, -1e18)
        energy = F.softmax(energy, dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination
