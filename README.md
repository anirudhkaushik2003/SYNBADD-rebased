# SynBADD

**Longitudinal 3D T1 MRI synthesis on ADNI using stationary velocity fields.**

Given a subject with longitudinal brain MRI scans (e.g., screening, month 6, month 12, ...), SynBADD predicts missing timepoints by learning temporal deformation dynamics through SVFs (stationary velocity fields), a VAE latent space, and attention-based temporal models.

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Train a single fold
synbadd train -c configs/base.yaml --fold 1

# Run full pipeline (train + infer + evaluate) across all folds
synbadd pipeline -c configs/base.yaml --folds 1,2,3,4,5
```

## Architecture

```
Input scans  -->  SVF computation  -->  VAE encode  -->  Temporal model  -->  VAE decode  -->  Generator  -->  Synthetic scan (.nii.gz)
     |                                                        |
     |              6 temporal model variants                  |
     |              (registry pattern — swap via config)       |
```

**Three inference methods** (swap with `--method`):

| Method | What it does | Needs checkpoints? |
|---|---|---|
| `synbadd` (default) | VAE encode -> temporal model -> VAE decode -> generator | Yes (all) |
| `avg_impute` | Average all available SVFs -> generator | Yes (generator only) |
| `forward_impute` | Copy most recent previous scan as prediction | No |

```bash
synbadd infer -c configs/base.yaml --fold 1 --method synbadd
synbadd infer -c configs/base.yaml --fold 1 --method avg_impute
synbadd infer -c configs/base.yaml --fold 1 --method forward_impute
```

## CLI Reference

```
synbadd train       -c <yaml> [--fold N] [--set key=val ...]
synbadd infer       -c <yaml> [--fold N] [--method synbadd|avg_impute|forward_impute] [--run-name NAME]
synbadd evaluate    -c <yaml> [--fold N] [--comparison noder|remind|ablation|avg_impute|forward_impute]
synbadd pipeline    -c <yaml> [--folds 1,2,3,4,5] [--stages train,infer,evaluate] [--skip-existing] [--infer-method ...]
synbadd manifest query --manifest-dir <dir> [--fold N] [--metric METRIC] [--top N]
synbadd manifest show <run-name> --manifest-dir <dir>
```

## Configuration

All hyperparameters, paths, and architecture choices live in YAML configs powered by OmegaConf.

```bash
# Use an experiment config that inherits from base
synbadd train -c configs/experiments/gated_cross_attn_PF.yaml --fold 1

# Override any value from the CLI
synbadd train -c configs/base.yaml --set model.temporal.type=CrossAttention --set training.n_epochs=50

# Hydra sweep for ablations
pip install -e ".[sweep]"
python sweep.py --multirun model.temporal.type=GatedCrossAttention,CrossAttention,BidirectionalTransformer data.fold=1,2,3,4,5
```

Experiment configs inherit from `base.yaml` using `_base_`:

```yaml
# configs/experiments/gated_cross_attn_PF.yaml
_base_: ../base.yaml
model:
  temporal:
    type: GatedCrossAttention
data:
  use_past_data: true
  use_future_data: true
```

## Adding a New Model

1. Create `synbadd/models/temporal/my_model.py`:

```python
from synbadd.models.registry import register_model
from synbadd.models.temporal.base import BaseTemporalModel

@register_model("MyNewTransformer")
class MyNewTransformer(BaseTemporalModel):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len, **kwargs):
        super().__init__()
        # ...

    def forward(self, x, missing_token_index):
        # ...
        return output
```

2. Set in config: `model.temporal.type: MyNewTransformer`

That's it. No other code changes needed.

## Project Structure

```
synbadd/
  cli.py                     # Click CLI entry point
  config/
    schema.py                # Frozen dataclasses (SynbaddConfig, ModelConfig, ...)
    loader.py                # YAML loading with _base_ inheritance + ${} interpolation
  models/
    registry.py              # @register_model decorator + build_model() factory
    spatial_ops.py           # SpatialTransformer, VecInt, ResizeTransform (single source)
    vae.py                   # VAE3D_UNetSkip
    generator.py             # SVFRegressorDiffeoGenerator
    discriminator.py         # DAFTDiscriminator
    temporal/                # 6 temporal model variants, each auto-registered
      base.py                # BaseTemporalModel ABC
      gated_cross_attention.py
      cross_attention.py
      cross_attention_gated.py
      bidirectional.py
      gated_transformer.py
      regressor.py
  data/
    base_dataset.py          # BaseSVFDataset ABC (shared CSV loading, fold splitting, masking)
    baseline_dataset.py      # BaselineSVFDataset (overrides _load_svf)
    piecewise_dataset.py     # PiecewiseSVFDataset (overrides _load_svf)
    transforms.py            # MONAI augmentation presets
    loader.py                # build_dataloaders() factory
  losses/
    similarity.py            # SSIM, PSNR, MSE, NRMSE
    regularization.py        # Smooth loss, gradient loss
    deformation.py           # Jacobian determinant, EPE, angular error
    lddmm.py                # VxmLDDMMLoss
    composite.py             # CompositeLoss assembled from config
  engine/
    trainer.py               # Training loop with early stopping, grad accumulation, W&B
    validator.py             # Validation and test loops
    inferrer.py              # Subject-wise .nii.gz export (synbadd / avg_impute / forward_impute)
    checkpoint.py            # CheckpointManager (backward-compatible with legacy naming)
  evaluation/
    base_evaluator.py        # BaseEvaluator ABC
    metrics.py               # Dice, volume, ROI extraction
    noder_comparison.py      # SynBADD vs NODER
    remind_comparison.py     # SynBADD vs ReMiND
    ablation_comparison.py   # Cross-model ablations
    avg_impute.py            # Average imputation baseline
    forward_impute.py        # Forward imputation baseline
  pipeline/
    orchestrator.py          # Pipeline: iterate folds x stages with --skip-existing
    stages.py                # TrainStage, InferStage, EvaluateStage
    manifest.py              # ExperimentManifest JSON tracking + ManifestStore queries
configs/
  base.yaml                  # All defaults
  experiments/               # Per-experiment overrides
scripts/                     # Preprocessing utilities (standalone)
tests/                       # pytest suite
```

## Pipeline

The pipeline command runs stages sequentially per fold:

```
Train (save checkpoints + manifest)
  -> Infer (load checkpoints, export .nii.gz + metrics CSV)
    -> Evaluate (compute Dice/volume/SSIM, save CSVs + plots)
```

Each stage checks prerequisites (do inputs exist?) and can skip if outputs already exist (`--skip-existing`), enabling crash recovery:

```bash
# If training finished but inference crashed, rerun — it picks up from infer
synbadd pipeline -c configs/base.yaml --folds 1,2,3,4,5 --skip-existing
```

Run only specific stages:

```bash
synbadd pipeline -c configs/base.yaml --folds 1 --stages infer,evaluate
```

## Experiment Manifest

Each completed run writes a JSON manifest tracking config, checkpoints, metrics, and W&B run ID. Query your best models:

```bash
synbadd manifest query --manifest-dir /scratch/anirudhkaushik/SYNBADD/manifests --fold 1 --metric best_val_loss --top 3
```

## Metrics

**Training**: SSIM loss, LDDMM loss, smooth loss, Jacobian regularization, EPE, angular error, optional adversarial loss

**Inference**: MSE, PSNR, SSIM, NRMSE per subject per timepoint

**Evaluation**: ROI Dice (hippocampus, amygdala, entorhinal cortex, ...), ROI volumes, volume difference (%), trajectory alignment

## Development

```bash
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check synbadd/ tests/

# Smoke test (2 epochs, no W&B)
synbadd train -c configs/base.yaml --fold 1 --set training.n_epochs=2 --set system.debug=true
```

## Legacy Code

The original (pre-refactor) codebase is preserved on the `v0-legacy` tag:

```bash
git checkout v0-legacy  # browse the original 400+ file repo
git checkout main       # back to clean codebase
```
