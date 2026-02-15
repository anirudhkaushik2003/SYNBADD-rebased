"""Tests for config loading, interpolation, and overrides."""

import os

import pytest

from synbadd.config.loader import load_config
from synbadd.config.schema import SynbaddConfig


@pytest.fixture
def base_yaml(tmp_path):
    """Write a minimal base.yaml to a temp dir."""
    content = """\
model:
  img_height: 128
  latent_dim: 512
  temporal:
    type: GatedCrossAttention
    num_heads: 8

training:
  n_epochs: 100
  batch_size: 4
  lr: 0.0001
  loss:
    ssim: 0.2
    lddmm: 0.5

data:
  fold: 1
  svf_type: baseline
  use_past_data: true
  use_future_data: true

paths:
  base_dir: /tmp/synbadd_test
  csv_file: ${paths.base_dir}/data.csv
  checkpoint_dir: ${paths.base_dir}/checkpoints

system:
  seed: 42
  debug: true

wandb:
  enabled: false
"""
    p = tmp_path / "base.yaml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def child_yaml(tmp_path, base_yaml):
    """Write a child config inheriting from base."""
    content = f"""\
_base_: {os.path.basename(base_yaml)}
model:
  temporal:
    type: CrossAttention
data:
  fold: 3
training:
  n_epochs: 50
"""
    p = tmp_path / "child.yaml"
    p.write_text(content)
    return str(p)


class TestConfigLoading:
    def test_load_defaults(self):
        """Loading without a file should give defaults."""
        cfg = load_config()
        assert isinstance(cfg, SynbaddConfig)
        assert cfg.model.img_height == 128
        assert cfg.data.fold == 1

    def test_load_from_yaml(self, base_yaml):
        cfg = load_config(base_yaml)
        assert cfg.model.img_height == 128
        assert cfg.model.temporal.type == "GatedCrossAttention"
        assert cfg.training.n_epochs == 100
        assert cfg.data.fold == 1
        assert cfg.system.debug is True

    def test_interpolation(self, base_yaml):
        """${paths.base_dir} should resolve in csv_file."""
        cfg = load_config(base_yaml)
        assert cfg.paths.csv_file == "/tmp/synbadd_test/data.csv"
        assert cfg.paths.checkpoint_dir == "/tmp/synbadd_test/checkpoints"

    def test_fold_override(self, base_yaml):
        cfg = load_config(base_yaml, fold=3)
        assert cfg.data.fold == 3

    def test_set_override(self, base_yaml):
        cfg = load_config(base_yaml, overrides=["training.n_epochs=50", "model.latent_dim=256"])
        assert cfg.training.n_epochs == 50
        assert cfg.model.latent_dim == 256

    def test_bool_override(self, base_yaml):
        cfg = load_config(base_yaml, overrides=["data.use_past_data=false"])
        assert cfg.data.use_past_data is False

    def test_base_inheritance(self, child_yaml):
        cfg = load_config(child_yaml)
        # Child overrides
        assert cfg.model.temporal.type == "CrossAttention"
        assert cfg.data.fold == 3
        assert cfg.training.n_epochs == 50
        # Inherited from parent
        assert cfg.model.img_height == 128
        assert cfg.model.latent_dim == 512
        assert cfg.training.loss.ssim == 0.2

    def test_config_frozen_fields(self, base_yaml):
        """Config should be a dataclass (immutable by convention)."""
        cfg = load_config(base_yaml)
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "training")
        assert hasattr(cfg, "data")
        assert hasattr(cfg, "paths")

    def test_paths_resolve_defaults(self):
        """Empty path fields should be filled from base_dir."""
        cfg = load_config()
        assert cfg.paths.svf_dir != ""
        assert cfg.paths.base_dir in cfg.paths.svf_dir

    def test_invalid_override_raises(self, base_yaml):
        with pytest.raises(ValueError, match="key=value"):
            load_config(base_yaml, overrides=["bad_override_no_equals"])
