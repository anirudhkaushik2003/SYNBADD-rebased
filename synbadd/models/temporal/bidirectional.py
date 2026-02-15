"""Bidirectional Transformer with past/future/missing token embeddings."""

import torch
import torch.nn as nn

from synbadd.models.registry import register_model
from synbadd.models.temporal.base import BaseTemporalModel, sinusoidal_embedding


@register_model("BidirectionalTransformer")
class TransformerBidirectional(BaseTemporalModel):

    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5,
                 use_positional_embeddings=True, use_bidirectional_embeddings=True,
                 **kwargs):
        super().__init__(input_dim, num_heads, num_layers, max_seq_len)
        self.use_positional_embeddings = use_positional_embeddings
        self.use_bidirectional_embeddings = use_bidirectional_embeddings

        if use_positional_embeddings:
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else:
            self.register_buffer('positional_embeddings',
                                 sinusoidal_embedding(max_seq_len, input_dim))

        # Bidirectional learnable embeddings
        self.past_embeddings = nn.Parameter(torch.zeros(1, input_dim))
        self.future_embeddings = nn.Parameter(torch.zeros(1, input_dim))
        self.missing_token_embeddings = nn.Parameter(torch.zeros(1, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            num_layers=num_layers, encoder_layer=encoder_layer,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, missing_token_index):
        batch_size, seq_len, input_dim = x.size()

        time = torch.arange(seq_len, device=x.device).unsqueeze(0)
        missing_index_expanded = missing_token_index.expand(-1, seq_len)

        past_mask = time < missing_index_expanded
        future_mask = time > missing_index_expanded
        missing_mask = time == missing_index_expanded

        embedding_token = torch.zeros_like(x)
        embedding_token += past_mask.unsqueeze(2) * self.past_embeddings
        embedding_token += future_mask.unsqueeze(2) * self.future_embeddings
        embedding_token += missing_mask.unsqueeze(2) * self.missing_token_embeddings

        if self.use_bidirectional_embeddings:
            x = x + embedding_token
        x = x + self.positional_embeddings[:, :seq_len, :]

        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)

        # Gather output at each sample's missing token index
        output = x.gather(1, missing_token_index.unsqueeze(2).expand(-1, -1, input_dim)).squeeze(1)
        return output
