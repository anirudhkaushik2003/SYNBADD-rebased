"""Piecewise SVF dataset: loads SVFs from a directory of .pt files."""

import os
import re

import numpy as np
import torch

from synbadd.data.base_dataset import BaseSVFDataset


class PiecewiseSVFDataset(BaseSVFDataset):
    """SVF dataset where SVFs are loaded from a directory of .pt files.

    Filenames follow the pattern: {subject_id}_{source_tp}_{target_tp}.pt
    SVF map: {subject_id -> {(source_tp, target_tp) -> svf_path}}.
    Each SVF is a piecewise (consecutive timepoint) displacement field.
    """

    FILENAME_PATTERN = re.compile(r"(\d+_S_\d+)_(\w+)_(\w+)\.pt")

    def __init__(self, cfg, csv_file, fold, subset="train"):
        self._svf_dir = cfg.paths.svf_dir
        self._warp_baseline = getattr(cfg.data, "warp_baseline", True)
        self._baseline_svf_dir = cfg.paths.svf_baseline_dir
        super().__init__(cfg, csv_file, fold, subset)

    def _build_svf_map(self):
        svf_dir = self._svf_dir
        if not os.path.isdir(svf_dir):
            raise ValueError(f"SVF directory not found: {svf_dir}")

        self.svf_map = {}
        for filename in os.listdir(svf_dir):
            if not filename.endswith(".pt"):
                continue
            match = self.FILENAME_PATTERN.match(filename)
            if not match:
                continue
            subject_id = match.group(1)
            source_tp = match.group(2)
            target_tp = match.group(3)

            if subject_id not in self.subjects:
                continue

            self.svf_map.setdefault(subject_id, {})[(source_tp, target_tp)] = (
                os.path.join(svf_dir, filename)
            )

    def _load_svf(self, subject_id, timepoint, subject_data):
        """Load piecewise SVF: previous_timepoint -> timepoint."""
        tp_idx = self.timepoints.index(timepoint)
        prev_tp = self.timepoints[tp_idx - 1] if tp_idx > 0 else None

        if (
            prev_tp
            and subject_id in self.svf_map
            and (prev_tp, timepoint) in self.svf_map[subject_id]
        ):
            svf_path = self.svf_map[subject_id][(prev_tp, timepoint)]
            # Piecewise SVFs are stored as [H, W, D, 3], permute to [3, H, W, D]
            return torch.load(svf_path, weights_only=False).permute(3, 0, 1, 2).cpu().detach().numpy()
        return None

    def _load_target_svf(self, subject_id, target_timepoint, subject_data):
        """Load target SVF based on warp_baseline setting."""
        if self._warp_baseline:
            # Load baseline-to-target SVF from baseline SVF directory
            filename = f"{subject_id}_sc_{target_timepoint}.pt"
            path = os.path.join(self._baseline_svf_dir, filename)
            if os.path.exists(path):
                return torch.load(path, weights_only=False).cpu().detach().numpy()
            return np.zeros(self.VOLUME_SHAPE)

        # Non-baseline: find nearest earlier timepoint with SVF to target
        target_tp_idx = self.timepoints.index(target_timepoint)

        # Search backwards for nearest available timepoint
        for idx in range(target_tp_idx - 1, -1, -1):
            source_tp = self.timepoints[idx]
            if (
                subject_id in self.svf_map
                and (source_tp, target_timepoint) in self.svf_map[subject_id]
            ):
                svf_path = self.svf_map[subject_id][(source_tp, target_timepoint)]
                return torch.load(svf_path, weights_only=False).permute(3, 0, 1, 2).cpu().detach().numpy()

        return np.zeros(self.VOLUME_SHAPE)
