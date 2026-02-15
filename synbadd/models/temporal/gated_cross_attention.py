"""Gated Cross-Attention Transformer: gating applied BEFORE cross-attention."""

import torch
import torch.nn as nn

from synbadd.models.registry import register_model
from synbadd.models.temporal.base import BaseTemporalModel, sinusoidal_embedding


@register_model("GatedCrossAttention")
class GatedCrossAttentionTransformer(BaseTemporalModel):

    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5,
                 use_positional_embeddings=True, **kwargs):
        super().__init__(input_dim, num_heads, num_layers, max_seq_len)
        self.use_positional_embeddings = use_positional_embeddings

        if use_positional_embeddings:
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else:
            self.register_buffer('positional_embeddings',
                                 sinusoidal_embedding(max_seq_len, input_dim))

        self.missing_time_embedding = nn.Embedding(max_seq_len, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers,
        )

        self.attention_fc = nn.Linear(input_dim, 1)
        self.gating_fc = nn.Linear(input_dim, 1)

    def forward(self, x, missing_token_index):
        batch_size, seq_len, input_dim = x.size()

        x = x + self.positional_embeddings[:, :seq_len, :]

        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)

        # Gated attention
        attn_scores = self.attention_fc(x).squeeze(-1)
        gating_scores = torch.sigmoid(self.gating_fc(x)).squeeze(-1)
        gated_scores = attn_scores * gating_scores
        gated_attn_weights = torch.softmax(gated_scores, dim=-1)

        x_gated = gated_attn_weights.unsqueeze(-1) * x

        # Cross attention with missing timepoint query
        query = self.missing_time_embedding(missing_token_index.squeeze(-1))
        attn_scores_cross = (x_gated * query.unsqueeze(1)).sum(dim=-1)
        attn_weights_cross = torch.softmax(attn_scores_cross, dim=-1)

        output = torch.bmm(attn_weights_cross.unsqueeze(1), x_gated).squeeze(1)
        return output
