"""Tests for pipeline stage prerequisites and output checking."""

import os

import pytest

from synbadd.config.schema import DataConfig, PathsConfig, SynbaddConfig
from synbadd.pipeline.stages import EvaluateStage, InferStage, TrainStage


@pytest.fixture
def tmp_cfg(tmp_path):
    """Create a config pointing to temp directories."""

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("Subject,Visit\n")

    vae_ckpt = tmp_path / "vae.pth"
    vae_ckpt.write_text("")

    vxm_ckpt = tmp_path / "vxm.pth"
    vxm_ckpt.write_text("")

    paths = PathsConfig(
        base_dir=str(tmp_path),
        csv_file=str(csv_file),
        vae_checkpoint=str(vae_ckpt),
        vxm_checkpoint=str(vxm_ckpt),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path / "outputs"),
    )

    cfg = SynbaddConfig(
        paths=paths,
        data=DataConfig(fold=1),
        run_name="test_run",
        project_name="test_project",
    )
    return cfg


class TestTrainStage:
    def test_prerequisites_pass(self, tmp_cfg):
        """Should pass when CSV, VAE, and VxM files exist."""
        stage = TrainStage(tmp_cfg, fold=1)
        stage.check_prerequisites()  # Should not raise

    def test_prerequisites_missing_csv(self, tmp_cfg):
        import dataclasses
        bad_paths = dataclasses.replace(tmp_cfg.paths, csv_file="/nonexistent/data.csv")
        bad_cfg = dataclasses.replace(tmp_cfg, paths=bad_paths)
        stage = TrainStage(bad_cfg, fold=1)
        with pytest.raises(FileNotFoundError, match="CSV"):
            stage.check_prerequisites()

    def test_prerequisites_missing_vae(self, tmp_cfg):
        import dataclasses
        bad_paths = dataclasses.replace(tmp_cfg.paths, vae_checkpoint="/nonexistent/vae.pth")
        bad_cfg = dataclasses.replace(tmp_cfg, paths=bad_paths)
        stage = TrainStage(bad_cfg, fold=1)
        with pytest.raises(FileNotFoundError, match="VAE"):
            stage.check_prerequisites()

    def test_outputs_exist_false(self, tmp_cfg):
        stage = TrainStage(tmp_cfg, fold=1)
        assert stage.outputs_exist() is False


class TestInferStage:
    def test_prerequisites_missing_checkpoint(self, tmp_cfg):
        stage = InferStage(tmp_cfg, fold=1)
        with pytest.raises(FileNotFoundError, match="Generator checkpoint"):
            stage.check_prerequisites()

    def test_outputs_exist_false(self, tmp_cfg):
        stage = InferStage(tmp_cfg, fold=1)
        assert stage.outputs_exist() is False

    def test_outputs_exist_true(self, tmp_cfg):
        # Create the expected output directory with a synthetic file
        output_dir = os.path.join(
            tmp_cfg.paths.output_dir,
            "SYNBADD_Qualitative_1",
            tmp_cfg.run_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "subj1_m12_synthetic.nii.gz"), "w") as f:
            f.write("")

        stage = InferStage(tmp_cfg, fold=1)
        assert stage.outputs_exist() is True


class TestEvaluateStage:
    def test_prerequisites_missing_outputs(self, tmp_cfg):
        stage = EvaluateStage(tmp_cfg, fold=1)
        with pytest.raises(FileNotFoundError, match="Inference outputs"):
            stage.check_prerequisites()

    def test_prerequisites_pass(self, tmp_cfg):
        # Create the expected output directory
        output_dir = os.path.join(
            tmp_cfg.paths.output_dir,
            "SYNBADD_Qualitative_1",
            tmp_cfg.run_name,
        )
        os.makedirs(output_dir, exist_ok=True)

        stage = EvaluateStage(tmp_cfg, fold=1)
        stage.check_prerequisites()  # Should not raise
