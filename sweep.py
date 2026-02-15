"""Optional Hydra entry point for grid-search sweeps.

This is a thin wrapper that translates Hydra configs into SynbaddConfig
and calls the same engine classes as the CLI. Use this for ablation sweeps:

    python sweep.py --multirun model.temporal.type=GatedCrossAttention,CrossAttention data.fold=1,2,3,4,5

For day-to-day use, prefer the CLI: synbadd train --config ...

Requires: pip install hydra-core  (or: pip install synbadd[sweep])
"""

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError(
        "Hydra is required for sweep.py. Install with: pip install hydra-core\n"
        "Or: pip install -e '.[sweep]'"
    )

from synbadd.config.loader import _dict_to_config


@hydra.main(version_base=None, config_path="configs", config_name="base")
def run(cfg: DictConfig) -> float:
    """Single sweep run. Returns validation loss for Hydra optimizers."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    synbadd_cfg = _dict_to_config(resolved)

    from synbadd.engine.trainer import Trainer

    trainer = Trainer(synbadd_cfg)
    result = trainer.train()
    return result["best_val_loss"]


if __name__ == "__main__":
    run()
