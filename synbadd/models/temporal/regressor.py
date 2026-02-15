"""Simple latent regressor (non-transformer fallback)."""

import torch.nn as nn

from synbadd.models.registry import register_model
from synbadd.models.temporal.base import BaseTemporalModel


@register_model("LatentRegressor")
class LatentRegressor(BaseTemporalModel):
    """Flattens the sequence and projects to a single latent vector."""

    def __init__(self, input_dim=512, num_heads=1, num_layers=1,
                 max_seq_len=5, **kwargs):
        super().__init__(input_dim, num_heads, num_layers, max_seq_len)

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_seq_len * input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x, missing_token_index=None):
        return self.net(x)
