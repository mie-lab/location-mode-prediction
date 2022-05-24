import torch.nn as nn
import numpy as np
import torch, math
from torch import Tensor

import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class AllEmbedding(nn.Module):
    def __init__(self, config, total_loc_num) -> None:
        super(AllEmbedding, self).__init__()
        # emberdding layers
        # location embedding
        self.emb_loc = nn.Embedding(total_loc_num, config.loc_emb_size)

        # time is in minutes, possible time for each day is 60 * 24 // 30
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            self.emb_time = nn.Embedding(60 * 24 // 30, config.time_emb_size)

        self.if_include_mode = config.if_embed_mode
        if self.if_include_mode:
            self.emb_mode = nn.Embedding(8, config.mode_emb_size)

    def forward(self, src, dict, device) -> Tensor:
        # embedding
        emb = self.emb_loc(src)

        if self.if_include_time:
            time = dict["time"].to(device)
            emb = torch.cat([emb, self.emb_time(time)], -1)
        if self.if_include_mode:
            mode = dict["mode"].to(device)
            emb = torch.cat([emb, self.emb_mode(mode)], -1)

        return emb


class Classifier(nn.Module):
    def __init__(self, config, total_loc_num) -> None:
        super(Classifier, self).__init__()
        self.Embedding = AllEmbedding(config, total_loc_num)

        # the input size to each layer
        self.d_input = config.loc_emb_size
        if config.if_embed_time:
            self.d_input = self.d_input + config.time_emb_size
        if config.if_embed_mode:
            self.d_input = self.d_input + config.mode_emb_size

        self.model_type = config.networkName

        if self.model_type == "transformer":
            # positional encoding
            self.pos_encoder = PositionalEncoding(self.d_input, config.dropout)
            self.model = Transformer(config, self.d_input)
            self.out_dim = self.d_input
        elif self.model_type == "rnn":
            self.model = RNN_Classifier(config, self.d_input)
            self.out_dim = config.hidden_size

            self.attention = config.attention
            if self.attention:
                self.attentionLayer = nn.MultiheadAttention(
                    embed_dim=self.out_dim,
                    num_heads=1,
                )
                self.norm = nn.LayerNorm(self.out_dim)

        else:
            assert False

        # the last fully connected layer
        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, config.user_emb_size)

            fc_dim = self.out_dim + config.user_emb_size
        else:
            fc_dim = self.out_dim

        self.fc_1 = nn.Linear(fc_dim, fc_dim)
        self.fc_loc = nn.Linear(fc_dim, total_loc_num)

        self.if_loss_mode = config.if_loss_mode
        if self.if_loss_mode:
            self.fc_mode = nn.Linear(fc_dim, 8)

        self.emb_dropout = nn.Dropout(p=0.1)
        self.fc_dropout = nn.Dropout(p=0.5)

        # init parameter
        self._init_weights()
        if self.model_type == "rnn":
            self._init_weights_rnn()

    def forward(self, src, tgt, src_context_dict, tgt_context_dict,  predict_length, device) -> Tensor:
        src_emb = self.Embedding(src, src_context_dict, device)
        tgt_emb = self.Embedding(tgt, tgt_context_dict, device)
        
        seq_len = src_context_dict["len"].to(device)

        # model
        if self.model_type == "transformer":
            # positional encoding, dropout performed inside
            src_emb = self.pos_encoder(src_emb * math.sqrt(self.d_input))
            tgt_emb = self.pos_encoder(tgt_emb * math.sqrt(self.d_input))
            # mask
            src_mask = torch.zeros((src.shape[0], src.shape[0])).type(torch.bool).to(device)
            src_padding_mask = (src == 0).transpose(0, 1).to(device)
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
            
            out = self.model(src=src_emb, tgt=tgt_emb, src_mask= src_mask, src_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask)

        elif self.model_type == "rnn":

            emb = self.dropout(emb)
            out, _ = self.model(emb)

            if self.attention:
                src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
                src_padding_mask = (src == 0).transpose(0, 1).to(device)
                attn_output, _ = self.attentionLayer(
                    out,
                    out,
                    out,
                    attn_mask=src_mask,
                    key_padding_mask=src_padding_mask,
                )
                # residual connection
                out = out + attn_output
                out = self.norm(out)
        # only take the last timestep
        # out = out.gather(
        #     0,
        #     seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        # ).squeeze(0)
        out = out[-predict_length:]

        # with fc output
        if self.if_embed_user:
            user = src_context_dict["user"].to(device)
            emb_user = self.emb_user(user).unsqueeze(0).expand(out.shape[0], -1, -1)
            
            out = torch.cat([out, emb_user], -1)
        out = self.emb_dropout(out)

        # residual
        out = out + self.fc_dropout(F.relu(self.fc_1(out)))

        if self.if_loss_mode:
            return self.fc_loc(out), self.fc_mode(out)
        else:
            return self.fc_loc(out)

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


class Transformer(nn.Module):
    def __init__(self, config, d_input) -> None:
        super(Transformer, self).__init__()
        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_input,
            nhead=config.nhead,
            activation="gelu",
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        encoder_norm = torch.nn.LayerNorm(d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
        )
        
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_input, 
            nhead=config.nhead, 
            activation="gelu",
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout)
        decoder_norm = torch.nn.LayerNorm(d_input)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=config.num_encoder_layers, norm=decoder_norm)


    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask) -> Tensor:
        """Forward pass of the network."""
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask)


class RNN_Classifier(nn.Module):
    """Baseline LSTM model."""

    def __init__(self, config, d_input):
        super(RNN_Classifier, self).__init__()
        RNNS = ["LSTM", "GRU"]
        self.bidirectional = False
        assert config.rnn_type in RNNS, "Use one of the following: {}".format(str(RNNS))
        rnn_cell = getattr(nn, config.rnn_type)  # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(
            d_input,
            hidden_size=config.hidden_size,
            num_layers=1,
            dropout=0.0,
            bidirectional=self.bidirectional,
        )

    def forward(self, input, hidden=None):
        """Forward pass of the network."""
        return self.rnn(input, hidden)
