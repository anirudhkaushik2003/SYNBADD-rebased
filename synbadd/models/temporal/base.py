"""Base class and shared utilities for temporal models."""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTemporalModel(nn.Module, ABC):
    """Base class for all temporal encoder models.

    All temporal models accept:
        x: [B, seq_len, dim] — sequence of latent vectors
        missing_token_index: [B, 1] — index of the missing/target timepoint
    And return:
        [B, dim] — predicted latent vector for the missing timepoint
    """

    def __init__(self, input_dim: int, num_heads: int, num_layers: int,
                 max_seq_len: int = 5, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

    @abstractmethod
    def forward(self, x: torch.Tensor, missing_token_index: torch.Tensor) -> torch.Tensor:
        ...


def sinusoidal_embedding(max_seq_len: int, dim: int) -> torch.Tensor:
    """Create sinusoidal positional embeddings (shared utility)."""
    pe = torch.zeros(max_seq_len, dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_seq_len, dim]
