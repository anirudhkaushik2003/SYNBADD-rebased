"""Minimal shared utilities."""

import random
from datetime import datetime

import numpy as np
import torch

from synbadd.config.schema import SynbaddConfig


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError(f"Boolean value expected, got: {v}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_run_name(cfg: SynbaddConfig) -> str:
    """Generate a unique run name from config, matching the original convention."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cfg.data.use_past_data and cfg.data.use_future_data:
        time_flag = "longitudinal"
    elif cfg.data.use_past_data:
        time_flag = "prev_only"
    elif cfg.data.use_future_data:
        time_flag = "future_only"
    else:
        time_flag = "cross_sectional"

    specials = cfg.specials if cfg.specials else f"{cfg.model.temporal.type}_Fold{cfg.data.fold}"

    return (
        f"SYNBADD_{cfg.model.img_height}_{cfg.training.effective_batch_size}_"
        f"{cfg.training.lr}_{cfg.training.aug_p}_{timestamp}_"
        f"{cfg.data.dataset}_{cfg.data.target_timepoint}_{time_flag}_{specials}"
    )


def get_timepoint_index(timepoint: str) -> int:
    """Map timepoint string to index."""
    mapping = {"sc": 0, "m06": 1, "m12": 2, "m18": 3, "m24": 4, "m36": 5}
    return mapping[timepoint]
