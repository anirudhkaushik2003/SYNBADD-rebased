"""Base SVF dataset with shared logic for all SVF-based sequence datasets."""

import random
from abc import ABC, abstractmethod

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from synbadd.data.transforms import eval_transforms


class BaseSVFDataset(Dataset, ABC):
    """Abstract base for longitudinal SVF sequence datasets.

    Shared logic: CSV loading, fold splitting, subject grouping, scan loading,
    target timepoint handling, past/future masking, __getitem__ output dict.

    Subclasses implement only:
        _build_svf_map()           -> populates self.svf_map
        _load_svf(subject_id, timepoint, subject_data) -> ndarray [3,H,W,D]
        _load_target_svf(subject_id, target_timepoint, subject_data) -> ndarray [3,H,W,D]
    """

    VOLUME_SHAPE = (3, 128, 128, 128)
    GROUP_MAP = {"CN": 0, "MCI": 1, "AD": 2}

    def __init__(self, cfg, csv_file, fold, subset="train"):
        self.cfg = cfg
        self.subset = subset
        self.timepoints = list(cfg.data.timepoints)
        self.target_timepoint = cfg.data.target_timepoint
        self.use_past_data = cfg.data.use_past_data
        self.use_future_data = cfg.data.use_future_data
        self.timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        # Transforms (scans always normalized; SVFs are not augmented)
        self.scan_transform = eval_transforms()

        # Load and filter CSV
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f"fold_{fold}"] == subset].reset_index(drop=True)

        # Merge val into train when use_val_dataset is False
        if subset == "train" and not cfg.data.use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f"fold_{fold}"] == "val"].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df["Acq Date"] = pd.to_datetime(self.df["Acq Date"])
        self.df = (
            self.df.sort_values(by=["Subject", "Visit", "Acq Date"])
            .drop_duplicates(subset=["Subject", "Visit"], keep="last")
        )

        # Group by subject
        self.subject_groups = self.df.groupby("Subject").apply(
            lambda x: x.sort_values(by="Acq Date")
        )
        self.subjects = list(self.subject_groups.index.levels[0])

        # Build SVF map (subclass-specific)
        self._build_svf_map()

        # Handle target timepoint modes: "all", "rand", or fixed
        self.subject_combinations = None
        self.subject_target_timepoints = None

        if self.target_timepoint == "all":
            self.subject_combinations = []
            for subject in self.subjects:
                available_visits = set(
                    self.df[self.df["Subject"] == subject]["Visit"].unique()
                )
                valid_tps = [
                    tp for tp in self.timepoints[1:] if tp in available_visits
                ]
                for tp in valid_tps:
                    self.subject_combinations.append((subject, tp))
        elif self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_targets()
        else:
            self.subject_target_timepoints = {
                s: self.target_timepoint for s in self.subjects
            }
            subjects_with_target = self.df[
                self.df["Visit"] == self.target_timepoint
            ]["Subject"].unique()
            self.df = self.df[
                self.df["Subject"].isin(subjects_with_target)
            ].reset_index(drop=True)
            self.subject_groups = self.df.groupby("Subject").apply(
                lambda x: x.sort_values(by="Acq Date")
            )
            self.subjects = list(self.subject_groups.index.levels[0])

        # File path <-> UID mapping (for inference / evaluation)
        self.file_path_to_id = {}
        for _, row in self.df.iterrows():
            uid = hash(row["File_Path"]) % (10**8)
            self.file_path_to_id[row["File_Path"]] = uid

    # ------------------------------------------------------------------
    # Abstract methods (subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_svf_map(self):
        """Populate self.svf_map from the SVF source (CSV or directory)."""

    @abstractmethod
    def _load_svf(self, subject_id, timepoint, subject_data):
        """Load a single SVF for a given subject/timepoint.

        Returns:
            ndarray of shape [3, H, W, D], or None if not found.
        """

    @abstractmethod
    def _load_target_svf(self, subject_id, target_timepoint, subject_data):
        """Load the ground-truth SVF for the target timepoint.

        Returns:
            ndarray of shape [3, H, W, D].
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _assign_random_targets(self):
        target_map = {}
        for subject in self.subjects:
            available = self.subject_groups.loc[subject]["Visit"].unique()
            valid = [tp for tp in self.timepoints[1:] if tp in available]
            if not valid:
                raise ValueError(f"No valid timepoints for subject {subject}")
            target_map[subject] = random.choice(valid)
        return target_map

    def load_scan(self, subject_id, visit="sc"):
        """Load a NIfTI scan, add channel dim, normalize."""
        visit_data = self.df[
            (self.df["Subject"] == subject_id) & (self.df["Visit"] == visit)
        ]
        if visit_data.empty:
            raise ValueError(f"No {visit} scan found for subject {subject_id}")
        file_path = visit_data.iloc[0]["File_Path"]
        scan = nib.load(file_path).get_fdata()
        if scan.ndim == 3:
            scan = np.expand_dims(scan, axis=0)
        return self.scan_transform(scan)

    def get_file_path_mapping(self):
        """Return uid -> file_path dict for reverse lookup."""
        return {uid: fp for fp, uid in self.file_path_to_id.items()}

    def __len__(self):
        if self.target_timepoint == "all":
            return len(self.subject_combinations)
        return len(self.subjects)

    def __getitem__(self, idx):
        # Resolve subject and target timepoint
        if self.target_timepoint == "all":
            subject_id, target_tp = self.subject_combinations[idx]
        else:
            subject_id = self.subjects[idx]
            target_tp = self.subject_target_timepoints[subject_id]

        subject_data = self.subject_groups.loc[subject_id]

        # Load scans
        latest_scan = self.load_scan(subject_id, visit="sc")
        target_scan = self.load_scan(subject_id, visit=target_tp)
        target_scan_path = subject_data[
            subject_data["Visit"] == target_tp
        ].iloc[0]["File_Path"]
        target_uid = self.file_path_to_id[target_scan_path]

        # Load target SVF (subclass-specific)
        target_svf = self._load_target_svf(subject_id, target_tp, subject_data)

        # Build SVF sequence and mask
        target_tp_index = self.timepoints.index(target_tp) - 1
        svfs = []
        mask = []

        for t_idx, timepoint in enumerate(self.timepoints):
            if timepoint == "sc":
                continue

            # Target timepoint is always masked
            if timepoint == target_tp:
                mask.append(0)
                svfs.append(np.zeros(self.VOLUME_SHAPE))
                continue

            # Past/future masking
            if not self.use_future_data:
                if t_idx > self.timepoints.index(target_tp):
                    if self.target_timepoint == "all":
                        mask.append(0)
                        svfs.append(np.zeros(self.VOLUME_SHAPE))
                        continue
                    else:
                        continue

            if not self.use_past_data:
                if t_idx < self.timepoints.index(target_tp):
                    if self.target_timepoint == "all":
                        mask.append(0)
                        svfs.append(np.zeros(self.VOLUME_SHAPE))
                        continue
                    else:
                        continue

            # Check if scan exists for this timepoint
            visit_row = subject_data[subject_data["Visit"] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros(self.VOLUME_SHAPE))
                continue

            # Load SVF (subclass-specific)
            svf = self._load_svf(subject_id, timepoint, subject_data)
            if svf is not None:
                mask.append(1)
                svfs.append(svf)
                if svf.shape != self.VOLUME_SHAPE:
                    raise ValueError(
                        f"SVF shape mismatch for {subject_id}, {timepoint}: {svf.shape}"
                    )
            else:
                mask.append(0)
                svfs.append(np.zeros(self.VOLUME_SHAPE))

        # Pack output tensors
        stacked_svf = torch.tensor(np.stack(svfs)).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()
        mask = torch.tensor(mask, dtype=torch.float32)

        class_label = torch.tensor(
            self.GROUP_MAP[subject_data.iloc[0]["Group"]]
        ).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]["Age"]).unsqueeze(0)
        target_tp_tensor = torch.tensor(self.timepoint_map[target_tp]).unsqueeze(0)
        latest_tp_tensor = torch.tensor(self.timepoint_map["sc"]).unsqueeze(0)
        target_tp_idx_tensor = torch.tensor(target_tp_index).unsqueeze(0)
        target_svf_tensor = torch.tensor(target_svf).float()
        target_uid_tensor = torch.tensor(target_uid).unsqueeze(0).int()

        return {
            "stacked_svf": stacked_svf,                # [T-1, 3, H, W, D]
            "latest_scan": latest_scan,                 # [1, H, W, D]
            "target_scan": target_scan,                 # [1, H, W, D]
            "class_label": class_label,                 # [1]
            "reg_label": reg_label,                     # [1]
            "target_timepoint": target_tp_tensor,       # [1]
            "latest_timepoint": latest_tp_tensor,       # [1]
            "mask": mask,                               # [T-1]
            "target_timepoint_index": target_tp_idx_tensor,  # [1]
            "target_svf": target_svf_tensor,            # [3, H, W, D]
            "target_uid": target_uid_tensor,            # [1]
        }
