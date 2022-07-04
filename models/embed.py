import torch
import torch.nn as nn
from torch import Tensor

import math


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


class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()

        self.emb_info = emb_info
        self.minute_size = 4
        hour_size = 24
        weekday = 7

        if self.emb_info == "all":
            self.minute_embed = nn.Embedding(self.minute_size, d_input)
            self.hour_embed = nn.Embedding(hour_size, d_input)
            self.weekday_embed = nn.Embedding(weekday, d_input)
        elif self.emb_info == "time":
            self.time_embed = nn.Embedding(self.minute_size * hour_size, d_input)
        elif self.emb_info == "weekday":
            self.weekday_embed = nn.Embedding(weekday, d_input)

    def forward(self, time, weekday):
        if self.emb_info == "all":
            hour = torch.div(time, self.minute_size, rounding_mode="floor")
            minutes = time % 4

            minute_x = self.minute_embed(minutes)
            hour_x = self.hour_embed(hour)
            weekday_x = self.weekday_embed(weekday)

            return hour_x + minute_x + weekday_x
        elif self.emb_info == "time":
            return self.time_embed(time)
        elif self.emb_info == "weekday":
            return self.weekday_embed(weekday)


class AllEmbedding(nn.Module):
    def __init__(self, d_input, config, if_pos_encoder=True, emb_info="all", emb_type="add") -> None:
        super(AllEmbedding, self).__init__()
        # emberdding layers
        self.d_input = d_input
        self.emb_type = emb_type

        # location embedding
        self.if_include_loc = config.if_embed_loc
        if self.if_include_loc:
            if self.emb_type == "add":
                self.emb_loc = nn.Embedding(config.total_loc_num, d_input)
            else:
                self.emb_loc = nn.Embedding(config.total_loc_num, d_input - config.time_emb_size)

        self.if_include_mode = config.if_embed_mode
        if self.if_include_mode:

            if config.dataset == "geolife":
                mode_size = 4
            else:
                mode_size = 8
            self.emb_mode = nn.Embedding(mode_size, d_input)

        # time is in minutes, possible time for each day is 60 * 24 // 30
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            if self.emb_type == "add":
                self.temporal_embedding = TemporalEmbedding(d_input, emb_info)
            else:
                self.temporal_embedding = TemporalEmbedding(config.time_emb_size, emb_info)

        self.if_pos_encoder = if_pos_encoder
        if self.if_pos_encoder:
            self.pos_encoder = PositionalEncoding(d_input, dropout=0.1)
        else:
            self.dropout = nn.Dropout(0.1)

    def get_modeEmbedding(self):
        return self.emb_mode

    def forward(self, src, context_dict) -> Tensor:
        if self.if_include_loc:
            emb = self.emb_loc(src)
        else:
            # use mode as the first embedding (only for predicting mode)
            emb = self.emb_mode(context_dict["mode"])

        # location and mode
        if self.if_include_loc and self.if_include_mode:
            emb = emb + self.emb_mode(context_dict["mode"])

        if self.if_include_time:
            if self.emb_type == "add":
                emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday"])
            else:
                emb = torch.cat([emb, self.temporal_embedding(context_dict["time"], context_dict["weekday"])], dim=-1)

        if self.if_pos_encoder:
            return self.pos_encoder(emb * math.sqrt(self.d_input))
        else:
            return self.dropout(emb)
