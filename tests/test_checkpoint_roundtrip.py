"""Tests for checkpoint save/load roundtrip."""


import pytest
import torch
import torch.nn as nn

from synbadd.engine.checkpoint import CheckpointManager, strip_module_prefix


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def ckpt_mgr(tmp_path):
    return CheckpointManager(str(tmp_path), "test_project", "test_run")


class TestCheckpointManager:
    def test_save_and_exists(self, ckpt_mgr):
        model = SimpleModel()
        ckpt_mgr.save({"generator": model})
        assert ckpt_mgr.exists("generator")

    def test_not_exists(self, ckpt_mgr):
        assert not ckpt_mgr.exists("generator")

    def test_save_load_roundtrip(self, ckpt_mgr):
        model = SimpleModel()
        x = torch.randn(3, 10)

        # Forward before save
        model.eval()
        with torch.no_grad():
            out_before = model(x)

        # Save
        ckpt_mgr.save({"generator": model})

        # Load into fresh model
        model2 = SimpleModel()
        ckpt_mgr.load({"generator": model2}, device="cpu")
        model2.eval()

        with torch.no_grad():
            out_after = model2(x)

        assert torch.allclose(out_before, out_after)

    def test_strip_module_prefix(self):
        """DataParallel wraps keys with 'module.' prefix — should be stripped."""
        state = {"module.fc.weight": torch.randn(5, 10), "module.fc.bias": torch.randn(5)}
        stripped = strip_module_prefix(state)
        for key in stripped:
            assert not key.startswith("module."), f"Key still has module prefix: {key}"
        assert "fc.weight" in stripped
        assert "fc.bias" in stripped

    def test_multiple_components(self, ckpt_mgr):
        """Save and load multiple components at once."""
        models = {
            "generator": SimpleModel(),
            "temporal": SimpleModel(),
        }
        ckpt_mgr.save(models)
        assert ckpt_mgr.exists("generator")
        assert ckpt_mgr.exists("temporal")

    def test_skip_none_models(self, ckpt_mgr):
        """None values in models_dict should be silently skipped."""
        ckpt_mgr.save({"generator": SimpleModel(), "discriminator": None})
        assert ckpt_mgr.exists("generator")
        assert not ckpt_mgr.exists("discriminator")
