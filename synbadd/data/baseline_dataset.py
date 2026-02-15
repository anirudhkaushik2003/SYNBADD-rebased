"""Baseline SVF dataset: loads SVFs from a CSV metadata file."""

import numpy as np
import pandas as pd
import torch

from synbadd.data.base_dataset import BaseSVFDataset


class BaselineSVFDataset(BaseSVFDataset):
    """SVF dataset where SVFs are loaded from a CSV file (svf_metadata.csv).

    The CSV has columns: Subject, Visit, SVF_Path.
    SVF map: {subject_id -> {visit -> svf_path}}.
    Each SVF is a baseline-to-timepoint displacement field of shape [3, H, W, D].
    """

    def __init__(self, cfg, csv_file, fold, subset="train"):
        self._svf_file = cfg.paths.svf_metadata
        super().__init__(cfg, csv_file, fold, subset)

    def _build_svf_map(self):
        svf_df = pd.read_csv(self._svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            self.svf_map.setdefault(row["Subject"], {})[row["Visit"]] = row["SVF_Path"]

    def _load_svf(self, subject_id, timepoint, subject_data):
        try:
            svf_path = self.svf_map[subject_id][timepoint]
            return torch.load(svf_path, weights_only=False).numpy()
        except KeyError:
            return None

    def _load_target_svf(self, subject_id, target_timepoint, subject_data):
        try:
            svf_path = self.svf_map[subject_id][target_timepoint]
            return torch.load(svf_path, weights_only=False).numpy()
        except KeyError:
            return np.zeros(self.VOLUME_SHAPE)
