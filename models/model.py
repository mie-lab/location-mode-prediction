import torch.nn as nn
import numpy as np
import torch, math
from torch import Tensor

import torch.nn.functional as F

from models.embed import AllEmbedding

class TransEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config)

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_input,
            nhead=config.nhead,
            activation="gelu",
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        encoder_norm = torch.nn.LayerNorm(self.d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
        )

        self.fc = FullyConnected(self.d_input, config, if_residual_layer=True)
        self.if_embed_next_mode = config.if_embed_next_mode

        # init parameter
        self._init_weights()

    def forward(self, src, context_dict, device, next_mode=None) -> Tensor:
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        if self.if_embed_next_mode:
            return self.fc(out, context_dict["user"], mode_emb=self.Embedding.get_modeEmbedding(), next_mode=next_mode)
        else:
            return self.fc(out, context_dict["user"])

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def _init_weights(self):
        """Initiate parameters in the transformer model."""
        # initrange = 0.1
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


class FullyConnected(nn.Module):
    def __init__(self, d_input, config, if_residual_layer=True):
        super(FullyConnected, self).__init__()
        # the last fully connected layer
        fc_dim = d_input

        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, config.user_emb_size)
            fc_dim = d_input + config.user_emb_size
        
        self.if_embed_next_mode = config.if_embed_next_mode
        if self.if_embed_next_mode:
            # mode number -> user_embed_size (add)
            self.next_mode_fc = nn.Linear(config.base_emb_size, config.user_emb_size)
            

        self.if_loss_mode = config.if_loss_mode
        if self.if_loss_mode:
            self.fc_mode = nn.Linear(fc_dim, 8)
        self.fc_loc = nn.Linear(fc_dim, config.total_loc_num)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            # the residual
            self.fc_1 = nn.Linear(fc_dim, fc_dim)
            self.norm_1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout = nn.Dropout(p=config.fc_dropout)

    def forward(self, out, user, mode_emb = None, next_mode=None) -> Tensor:

        # with fc output
        if self.if_embed_user:
            emb = self.emb_user(user)

            if self.if_embed_next_mode:
                emb += self.next_mode_fc(mode_emb(next_mode))

            out = torch.cat([out, emb], -1)
        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

        if self.if_loss_mode:
            return self.fc_loc(out), self.fc_mode(out)
        else:
            return self.fc_loc(out)


# class FullyConnectedMode(nn.Module):
#     def __init__(self, d_input, config, if_residual_layer=True):
#         super(FullyConnectedMode, self).__init__()
#         # the last fully connected layer
#         self.if_embed_user = config.if_embed_user
#         if self.if_embed_user:
#             self.emb_user = nn.Embedding(config.total_user_num, config.user_emb_size)
#             fc_dim = d_input + config.user_emb_size
#         else:
#             fc_dim = d_input

#         self.fc_mode = nn.Linear(fc_dim, 8)

        
#         if 
#         self.fc_loc = nn.Linear(fc_dim, config.total_loc_num)
#         self.emb_dropout = nn.Dropout(p=0.1)

#         self.if_residual_layer = if_residual_layer
#         if self.if_residual_layer:
#             # the residual
#             self.fc_1 = nn.Linear(fc_dim, fc_dim)
#             self.norm_1 = nn.BatchNorm1d(fc_dim)
#             self.fc_dropout = nn.Dropout(p=config.fc_dropout)

#     def forward(self, out, user) -> Tensor:

#         # with fc output
#         if self.if_embed_user:
#             emb_user = self.emb_user(user)
#             out = torch.cat([out, emb_user], -1)
#         out = self.emb_dropout(out)

#         # residual
#         if self.if_residual_layer:
#             out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

#         if self.if_loss_mode:
#             return self.fc_loc(out), self.fc_mode(out)
#         else:
#             return self.fc_mode(out)