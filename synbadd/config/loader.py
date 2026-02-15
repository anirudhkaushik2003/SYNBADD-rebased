"""Config loading with YAML inheritance, interpolation, and CLI overrides.

Supports:
- _base_ key for config inheritance (child values override parent)
- ${paths.base_dir} style variable interpolation (via OmegaConf)
- --set key=val CLI overrides (dot-notation paths)
"""

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from synbadd.config.schema import (
    BaselinesConfig,
    DataConfig,
    DiscriminatorConfig,
    EvalConfig,
    GeneratorConfig,
    LossConfig,
    ModelConfig,
    PathsConfig,
    SynbaddConfig,
    SystemConfig,
    TemporalConfig,
    TrainingConfig,
    VAEConfig,
    WandbConfig,
)


def _resolve_base(cfg_dict: dict, config_dir: Path) -> dict:
    """Recursively resolve _base_ inheritance."""
    if "_base_" not in cfg_dict:
        return cfg_dict

    base_path = config_dir / cfg_dict.pop("_base_")
    base_cfg = OmegaConf.to_container(OmegaConf.load(base_path), resolve=False)
    base_cfg = _resolve_base(base_cfg, base_path.parent)

    # Deep merge: child overrides parent
    merged = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(base_cfg), OmegaConf.create(cfg_dict)),
        resolve=False,
    )
    return merged


def _apply_overrides(cfg: DictConfig, overrides: List[str]) -> DictConfig:
    """Apply --set key=val overrides using dot notation."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, val = override.split("=", 1)

        # Auto-cast types
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass

        OmegaConf.update(cfg, key, val)
    return cfg


def _build_sub_config(cls, data: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def _dict_to_config(d: dict) -> SynbaddConfig:
    """Convert a resolved dict into typed SynbaddConfig."""
    model_d = d.get("model", {})
    gen_cfg = _build_sub_config(GeneratorConfig, model_d.get("generator", {}))
    temp_cfg = _build_sub_config(TemporalConfig, model_d.get("temporal", {}))
    disc_cfg = _build_sub_config(DiscriminatorConfig, model_d.get("discriminator", {}))
    vae_cfg = _build_sub_config(VAEConfig, model_d.get("vae", {}))
    model_cfg = ModelConfig(
        img_height=model_d.get("img_height", 128),
        latent_dim=model_d.get("latent_dim", 512),
        c_dim=model_d.get("c_dim", 3),
        in_channels=model_d.get("in_channels", 3),
        generator=gen_cfg,
        temporal=temp_cfg,
        discriminator=disc_cfg,
        vae=vae_cfg,
    )

    loss_d = d.get("training", {}).get("loss", {})
    loss_cfg = _build_sub_config(LossConfig, loss_d)

    train_d = d.get("training", {})
    betas = train_d.get("betas", [0.5, 0.999])
    if isinstance(betas, list):
        betas = tuple(betas)
    train_cfg = TrainingConfig(
        n_epochs=train_d.get("n_epochs", 200),
        batch_size=train_d.get("batch_size", 6),
        effective_batch_size=train_d.get("effective_batch_size", 6),
        lr=train_d.get("lr", 5e-5),
        betas=betas,
        patience=train_d.get("patience", 5),
        early_stopping_patience=train_d.get("early_stopping_patience", 7),
        min_lr=train_d.get("min_lr", 1e-6),
        aug_p=train_d.get("aug_p", 0.0),
        loss=loss_cfg,
    )

    data_d = d.get("data", {})
    data_cfg = _build_sub_config(DataConfig, data_d)

    paths_d = d.get("paths", {})
    baselines_d = paths_d.pop("baselines", {})
    baselines_cfg = _build_sub_config(BaselinesConfig, baselines_d) if baselines_d else BaselinesConfig()
    import dataclasses as _dc
    paths_field_names = {f.name for f in _dc.fields(PathsConfig)} - {"baselines"}
    paths_filtered = {k: v for k, v in paths_d.items() if k in paths_field_names}
    paths_cfg = PathsConfig(**paths_filtered, baselines=baselines_cfg)

    system_d = d.get("system", {})
    system_cfg = _build_sub_config(SystemConfig, system_d)

    wandb_d = d.get("wandb", {})
    wandb_cfg = _build_sub_config(WandbConfig, wandb_d)

    eval_d = d.get("eval", {})
    eval_cfg = _build_sub_config(EvalConfig, eval_d)

    return SynbaddConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        paths=paths_cfg,
        system=system_cfg,
        wandb=wandb_cfg,
        eval=eval_cfg,
        run_name=d.get("run_name", ""),
        project_name=d.get("project_name", ""),
        specials=d.get("specials", ""),
    )


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    fold: Optional[int] = None,
) -> SynbaddConfig:
    """Load config from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        overrides: List of "key=val" strings for dot-notation overrides.
        fold: Convenience override for data.fold.

    Returns:
        Frozen SynbaddConfig instance.
    """
    if config_path is not None:
        config_file = Path(config_path)
        raw = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
        raw = _resolve_base(raw, config_file.parent)
        cfg = OmegaConf.create(raw)
    else:
        cfg = OmegaConf.structured(SynbaddConfig)

    # Apply CLI overrides
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    # Convenience fold override
    if fold is not None:
        OmegaConf.update(cfg, "data.fold", fold)

    # Resolve interpolations
    resolved = OmegaConf.to_container(cfg, resolve=True)

    return _dict_to_config(resolved)
