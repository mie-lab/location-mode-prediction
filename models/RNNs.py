import torch
import torch.nn as nn
from models.embed import AllEmbedding
from models.model import FullyConnected


class RNNs(nn.Module):
    """Baseline LSTM model."""

    def __init__(self, config):
        super(RNNs, self).__init__()
        self.attention = config.attention
        self.d_input = config.loc_emb_size
        self.out_dim = config.hidden_size

        if self.attention:
            self.Embedding = AllEmbedding(self.d_input, config, if_pos_encoder=False, emb_info="weekday")
        else:
            self.Embedding = AllEmbedding(self.d_input, config, if_pos_encoder=False, emb_info="all")

        self.model = RNN_Classifier(self.d_input, config)
        
        
        if self.attention:
            self.attentionLayer = nn.MultiheadAttention(
                embed_dim=self.out_dim,
                num_heads=1,
            )
            self.norm = nn.LayerNorm(self.out_dim)

        self.fc = FullyConnected(self.out_dim, config, if_residual_layer=True)

        self._init_weights_rnn()

    def forward(self, src, context_dict, device):
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # model
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
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        return self.fc(out, context_dict["user"])

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

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


class RNN_Classifier(nn.Module):
    """Baseline LSTM model."""

    def __init__(self, d_input, config):
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
