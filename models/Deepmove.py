import torch.nn as nn
import numpy as np
import torch
from torch import Tensor

from models.embed import AllEmbedding
from models.RNNs import RNN_Classifier
from models.model import FullyConnected

import torch.nn.functional as F


class Deepmove(nn.Module):
    def __init__(self, config) -> None:
        super(Deepmove, self).__init__()
        # the input size to each layer
        self.d_input = config.loc_emb_size
        self.out_dim = config.hidden_size*2

        self.Embedding = AllEmbedding(self.d_input, config, if_pos_encoder=False, emb_info="time")
        self.rnn_encoder = RNN_Classifier( self.d_input, config)
        self.rnn_decoder = RNN_Classifier( self.d_input, config)

        self.attn = Attention()

        self.fc = FullyConnected(self.out_dim, config, if_residual_layer=True)

        # init parameter
        self._init_weights_rnn()

    def forward(self, src, src_dict, device) -> Tensor:

        hist, curr = src
        hist_context, curr_context = src_dict

        # length of each batch
        history_batch_len = hist_context["len"]
        curr_batch_len = curr_context["len"]

        # embedding
        history_emb = self.Embedding(hist, hist_context)
        curr_emb = self.Embedding(curr, curr_context)

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

        # with fc output
        return self.fc(out, curr_context["user"])

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
