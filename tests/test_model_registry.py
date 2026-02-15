"""Tests for model registry and model instantiation."""

import pytest
import torch

# Trigger all registrations
import synbadd.models  # noqa: F401
from synbadd.models.registry import build_model, list_models


class TestModelRegistry:
    def test_list_models_not_empty(self):
        models = list_models()
        assert len(models) > 0

    def test_all_temporal_models_registered(self):
        expected = [
            "GatedTransformerModel",
            "GatedCrossAttention",
            "CrossAttentionTransformer",
            "CrossAttentionGatedTransformer",
            "BidirectionalTransformer",
            "LatentRegressor",
        ]
        available = list_models()
        for name in expected:
            assert name in available, f"Model '{name}' not registered. Available: {available}"

    @pytest.mark.parametrize("model_name", [
        "GatedTransformerModel",
        "GatedCrossAttention",
        "CrossAttentionTransformer",
        "CrossAttentionGatedTransformer",
        "BidirectionalTransformer",
    ])
    def test_temporal_model_forward(self, model_name):
        """Each temporal model should accept (B, T, D) input and return (B, D)."""
        input_dim = 512
        max_seq_len = 5
        model = build_model(
            model_name,
            input_dim=input_dim,
            num_heads=8,
            num_layers=2,
            max_seq_len=max_seq_len,
        )
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, max_seq_len, input_dim)
        # Models expect a tensor for missing_token_index, shape (B,) or (B, 1)
        missing_idx = torch.tensor([[2], [3]])

        with torch.no_grad():
            out = model(x, missing_idx)

        assert out.shape == (batch_size, input_dim), f"Expected ({batch_size}, {input_dim}), got {out.shape}"

    def test_latent_regressor_forward(self):
        """LatentRegressor has different constructor signature."""
        model = build_model(
            "LatentRegressor",
            input_dim=512,
            max_seq_len=5,
        )
        model.eval()

        x = torch.randn(2, 5, 512)
        missing_idx = torch.tensor([[2], [3]])
        with torch.no_grad():
            out = model(x, missing_idx)

        assert out.shape == (2, 512)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            build_model("NonExistentModel", input_dim=128)
