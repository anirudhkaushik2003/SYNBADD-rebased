"""Gated Transformer with attention + gating pooling."""

import torch
import torch.nn as nn

from synbadd.models.registry import register_model
from synbadd.models.temporal.base import BaseTemporalModel


@register_model("GatedTransformerModel")
class GatedTransformerModel(BaseTemporalModel):

    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5, **kwargs):
        super().__init__(input_dim, num_heads, num_layers, max_seq_len)

        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers,
        )

        self.attention_fc = nn.Linear(input_dim, 1)
        self.gating_fc = nn.Linear(input_dim, 1)

    def forward(self, x, missing_token_index=None):
        batch_size, seq_len, input_dim = x.size()

        x = x + self.positional_embeddings[:, :seq_len, :]

        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)

        attn_scores = self.attention_fc(x).squeeze(-1)
        gating_scores = torch.sigmoid(self.gating_fc(x)).squeeze(-1)
        gated_attn_scores = attn_scores * gating_scores
        attn_weights = torch.softmax(gated_attn_scores, dim=1)

        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled_output
