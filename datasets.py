# =============================================================================
# File: datasets.py
# Purpose: Dataset and dataloader helper classes for ADNI/SVF/sequence settings used throughout training and evaluation.
# Prerequisites: Requires ADNI metadata and SVF resources in /scratch paths referenced by entry scripts.
# Usually Run Next: Instantiated by train/test/eval scripts to build fold-specific subsets.
# Model Components Used: No model definitions; supplies data tensors consumed by generator/VAE/transformer pipelines.
# =============================================================================

import numpy as np

from torch.utils.data import Dataset
from monai.data.dataloader import DataLoader as MonaiDataLoader
import nibabel as nib
import pandas as pd
import numpy as np
import torch

import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import monai.transforms as t
from monai.transforms import *

from multiprocessing import Pool

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import scipy.ndimage as ndi

from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
import os
import re


class PiecewiseSVFUltraMaskedSequenceDataset(Dataset):
    def __init__(self, opt, csv_file, svf_dir, fold, subset='train', aug_p=0.8,
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18',
                 log_missing_scans=False, use_val_dataset=False, warp_baseline=True):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.opt = opt

        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        self.subset = subset
        self.aug_p = aug_p
        self.log_missing_scans = log_missing_scans
        self.warp_baseline = warp_baseline

        # Grouping subjects
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        # Load SVF mapping from directory: {subject -> {(source, target) -> svf_path}}
        self.svf_map = self._load_svfs_from_directory(svf_dir)
        
        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        elif self.target_timepoint == "all":
            # For each subject, create multiple entries with different target timepoints
            self.subject_combinations = []
            for subject in self.subjects:
                # Get the actually available timepoints for this subject
                available_visits = set(self.df[self.df['Subject'] == subject]['Visit'].unique())
                valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]  # Skip 'sc'
                
                # Only include timepoints that have actual scans available
                for target_tp in valid_timepoints:
                    # Check if target timepoint scan actually exists
                    if target_tp in available_visits:
                        self.subject_combinations.append((subject, target_tp))
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            self.subjects = list(self.subject_groups.index.levels[0])

        # create a mapping between file path and id so we can return the id in the dataloader and reconvert it to filepath string
        self.file_path_to_id = {}
        for i, row in self.df.iterrows():
            file_path = row['File_Path']
            subject_id = row['Subject']
            visit = row['Visit']
            # generate an int uid
            uid = hash(file_path) % (10 ** 8)
            self.file_path_to_id[file_path] = uid

    def _load_svfs_from_directory(self, svf_dir):
        """Load SVF files from directory and organize them by subject and timepoint transition"""
        svf_map = {}
        
        if not os.path.isdir(svf_dir):
            raise ValueError(f"SVF directory not found: {svf_dir}")
        
        # Regex pattern to extract subject ID and timepoints from filenames
        pattern = r'(\d+_S_\d+)_(\w+)_(\w+)\.pt'
        
        for filename in os.listdir(svf_dir):
            if filename.endswith('.pt'):
                match = re.match(pattern, filename)
                if match:
                    subject_id = match.group(1)
                    source_timepoint = match.group(2)
                    target_timepoint = match.group(3)
                    
                    # Skip subjects not in our filtered dataset
                    if subject_id not in self.subjects:
                        continue
                    
                    if subject_id not in svf_map:
                        svf_map[subject_id] = {}
                    
                    # Store mapping: (source_timepoint, target_timepoint) -> svf_path
                    svf_map[subject_id][(source_timepoint, target_timepoint)] = os.path.join(svf_dir, filename)
        
        return svf_map

    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map

    def __len__(self):
        if self.target_timepoint == "all":
            return len(self.subject_combinations)
        return len(self.subjects)

    def load_scan(self, subject_id, visit='sc'):
        visit_data = self.df[(self.df['Subject'] == subject_id) & (self.df['Visit'] == visit)]
        if visit_data.empty:
            raise ValueError(f"No {visit} scan found for subject {subject_id}")
        file_path = visit_data.iloc[0]['File_Path']
        nii = nib.load(file_path)
        scan = nii.get_fdata()
        if scan.ndim == 3:
            scan = np.expand_dims(scan, axis=0)  # add channel dimension

        # apply transform
        scan = self.test_transforms(scan)
        return scan
    
    def get_file_path_mapping(self):
        uid_to_file_path = {}
        for file_path, uid in self.file_path_to_id.items():
            if uid not in uid_to_file_path:
                uid_to_file_path[uid] = file_path
        return uid_to_file_path

    def __getitem__(self, idx):

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        if self.target_timepoint == "all":
            subject_id, target_timepoint = self.subject_combinations[idx]
        else:
            subject_id = self.subjects[idx]
            target_timepoint = self.subject_target_timepoints[subject_id]

        subject_data = self.subject_groups.loc[subject_id]
        
        mask = []

        # Load screening scan as latest_scan
        latest_scan = self.load_scan(subject_id, visit='sc')

        # Load target scan
        target_scan = self.load_scan(subject_id, visit=target_timepoint)
        target_scan_path = subject_data[subject_data['Visit'] == target_timepoint].iloc[0]['File_Path']
        target_uid = self.file_path_to_id[target_scan_path]

        # SVF stack for all timepoints except sc
        svfs = []
        
        # Find the target SVF and latest scan based on warp_baseline parameter
        target_timepoint_index = self.timepoints.index(target_timepoint) - 1
        
        # Initialize target_svf to zeros
        target_svf = np.zeros((3, 128, 128, 128))
        
        if self.warp_baseline:
            # Use baseline scan as latest_scan (already loaded above)
            latest_scan_timepoint = timepoint_map['sc']
            
            # Load baseline-to-target SVF from hardcoded directory
            baseline_svf_dir = "/scratch/anirudhkaushik/svf_outputs"
            baseline_svf_filename = f"{subject_id}_sc_{target_timepoint}.pt"
            baseline_svf_path = os.path.join(baseline_svf_dir, baseline_svf_filename)
            
            if os.path.exists(baseline_svf_path):
                target_svf = torch.load(baseline_svf_path).cpu().detach().numpy()
        else:
            # Find nearest available timepoint before target_timepoint
            nearest_tp = None
            nearest_tp_idx = None
            
            # Start from the timepoint just before target and search backwards
            for idx in range(self.timepoints.index(target_timepoint) - 1, -1, -1):
                candidate_tp = self.timepoints[idx]
                # Check if scan exists for this timepoint
                if candidate_tp == 'sc' or any(subject_data['Visit'] == candidate_tp):
                    nearest_tp = candidate_tp
                    nearest_tp_idx = idx
                    break
            
            # If no earlier timepoint found, use sc as fallback
            if nearest_tp is None:
                nearest_tp = 'sc'
                nearest_tp_idx = 0
                
            # Load the nearest scan as latest_scan
            latest_scan = self.load_scan(subject_id, visit=nearest_tp)
            latest_scan_timepoint = timepoint_map[nearest_tp]
            
            # Try to find SVF from nearest timepoint to target
            svf_found = False
            
            # First check direct SVF from nearest to target
            if subject_id in self.svf_map and (nearest_tp, target_timepoint) in self.svf_map[subject_id]:
                target_svf_path = self.svf_map[subject_id][(nearest_tp, target_timepoint)]
                target_svf = torch.load(target_svf_path).permute(3, 0, 1, 2).cpu().detach().numpy()
                svf_found = True
            else:
                # If direct SVF not found, look for any SVF to target, starting from timepoints before nearest
                for idx in range(nearest_tp_idx - 1, -1, -1):
                    source_tp = self.timepoints[idx]
                    if subject_id in self.svf_map and (source_tp, target_timepoint) in self.svf_map[subject_id]:
                        target_svf_path = self.svf_map[subject_id][(source_tp, target_timepoint)]
                        target_svf = torch.load(target_svf_path).permute(3, 0, 1, 2).cpu().detach().numpy()
                        svf_found = True
                        break
        
        # Process each timepoint (except screening)
        for t, timepoint in enumerate(self.timepoints):
            if timepoint == 'sc':
                continue
            
            if timepoint == target_timepoint:
                # add mask
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))   
                continue
            
            # if not self.opt.use_future_data then skip scans after target_timepoint
            # if self.target_timepoint == "all" then add a mask instead of skipping
            if self.opt.use_future_data == False:
                if self.target_timepoint == "all":
                    if t > self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t > self.timepoints.index(target_timepoint):
                        continue
            
            if self.opt.use_past_data == False:
                if self.target_timepoint == "all":
                    if t < self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t < self.timepoints.index(target_timepoint):
                        continue
            
            # Check if scan exists for this timepoint
            visit_row = subject_data[subject_data['Visit'] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
                continue
            
            # For this timepoint, find the previous timepoint in the sequence
            tp_idx = self.timepoints.index(timepoint)
            prev_tp_idx = tp_idx - 1
            prev_tp = self.timepoints[prev_tp_idx] if prev_tp_idx >= 0 else None
            
            # Try to find the piecewise SVF
            if prev_tp and subject_id in self.svf_map and (prev_tp, timepoint) in self.svf_map[subject_id]:
                mask.append(1)
                svf_path = self.svf_map[subject_id][(prev_tp, timepoint)]
                svf = torch.load(svf_path).permute(3, 0, 1, 2).cpu().detach().numpy()  # [3, H, W, D]
                svfs.append(svf)
                if svf.shape != (3, 128, 128, 128):
                    raise ValueError(f"SVF shape mismatch for subject {subject_id}, timepoint {timepoint}: {svf.shape}")
            else:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
        
        stacked_svf = torch.tensor(np.stack(svfs)).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()
        mask = torch.tensor(mask, dtype=torch.float32)

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map['sc']).unsqueeze(0)
        target_timepoint_index = torch.tensor(target_timepoint_index).unsqueeze(0)
        target_svf = torch.tensor(target_svf).float()
        target_uid = torch.tensor(target_uid).unsqueeze(0).int()

        return {
            "stacked_svf": stacked_svf,                       # [T-1, 3, H, W, D]
            "latest_scan": latest_scan,                       # [1, H, W, D]
            "target_scan": target_scan,                       # [1, H, W, D]
            "class_label": class_label,                       # [1]
            "reg_label": reg_label,                           # [1]
            "target_timepoint": target_timepoint_tensor,      # [1]
            "latest_timepoint": latest_scan_timepoint,        # [1]
            "mask": mask,                                     # [T]
            "target_timepoint_index": target_timepoint_index, # [1]
            "target_svf": target_svf,                         # [3, H, W, D]
            "target_uid": target_uid                          # [1]
        }

class TimeAwareVAEDataset(Dataset):
    def __init__(self, svf_file, csv_file, fold, subset='train',
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'],
                 synthetic_dir=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']) \
                         .drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.timepoint_mapping = {tp: i for i, tp in enumerate(timepoints)}
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        svf_df = pd.read_csv(svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            subj = row['Subject']
            visit = row['Visit']
            svf_path = row['SVF_Path']
            self.svf_map.setdefault(subj, {})[visit] = svf_path

        self.samples = []
        for subject in self.subjects:
            visits = self.svf_map.get(subject, {})
            for timepoint in self.timepoints:
                if timepoint != 'sc' and timepoint in visits:
                    self.samples.append((subject, timepoint, False))

        if synthetic_dir is not None and os.path.isdir(synthetic_dir) and subset == 'train':
            synthetic_paths = sorted([
                os.path.join(synthetic_dir, f)
                for f in os.listdir(synthetic_dir)
                if f.endswith(".pt")
            ])
            for path in synthetic_paths:
                filename = os.path.basename(path)
                target_month = self._extract_target_month(filename)
                if target_month in self.timepoint_mapping:
                    self.samples.append((None, path, True, target_month))
                else:
                    self.samples.append((None, path, True, 'm24'))  # Default if extraction fails

    def _extract_target_month(self, filename):
        pattern = r'\d+_S_\d+_(\w+)_(\w+)(?:_aug\d+)?\.pt'
        match = re.match(pattern, filename)
        if match:
            return match.group(2)  # target month is the second capturing group
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.samples[idx]
        
        is_synthetic = sample_data[2]
        
        if is_synthetic:
            subject_id, path, _, target_month = sample_data
            svf = torch.load(path).float()
            timepoint = target_month
        else:
            subject_id, timepoint, _ = sample_data
            svf_path = self.svf_map[subject_id][timepoint]
            svf = torch.load(svf_path).float()
        
        return svf, timepoint

class VAEdeformationDataset(Dataset):
    def __init__(self, svf_file, csv_file, fold, subset='train',
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'],
                 synthetic_dir=None):
        """
        Dataset that loads real SVFs from mapping file and optionally synthetic SVFs from a directory.
        """
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']) \
                         .drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        # Load real SVF paths
        svf_df = pd.read_csv(svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            subj = row['Subject']
            visit = row['Visit']
            svf_path = row['SVF_Path']
            self.svf_map.setdefault(subj, {})[visit] = svf_path

        # Build real sample list
        self.samples = []
        for subject in self.subjects:
            visits = self.svf_map.get(subject, {})
            for timepoint in self.timepoints:
                if timepoint != 'sc' and timepoint in visits:
                    self.samples.append((subject, timepoint, False))  # False = real sample

        # Load synthetic SVF file paths (if provided)
        if synthetic_dir is not None and os.path.isdir(synthetic_dir) and subset == 'train':
            synthetic_paths = sorted([
                os.path.join(synthetic_dir, f)
                for f in os.listdir(synthetic_dir)
                if f.endswith(".pt")
            ])
            for path in synthetic_paths:
                self.samples.append((None, path, True))  # True = synthetic sample
        else:
            synthetic_paths = []

        print(f"Loaded {len(self.samples)} total SVFs ({len(self.samples) - len(synthetic_paths)} real, {len(synthetic_paths)} synthetic)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_id, timepoint_or_path, is_synthetic = self.samples[idx]

        if is_synthetic:
            svf = torch.load(timepoint_or_path).float()  # timepoint_or_path = synthetic path
        else:
            try:
                svf_path = self.svf_map[subject_id][timepoint_or_path]
                svf = torch.load(svf_path).float()
            except KeyError:
                raise ValueError(f"SVF not found for subject {subject_id}, timepoint {timepoint_or_path}")

        return svf

class SVFRobustnessSequenceDataset(Dataset):
    def __init__(self, opt, csv_file, svf_file, fold, subset='train', aug_p=0.8,
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18',
                 log_missing_scans=False, use_val_dataset=False, num_scans=2, 
                 future_impute_masking=False, mask_window=False):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.opt = opt

        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        self.subset = subset
        self.aug_p = aug_p
        self.log_missing_scans = log_missing_scans
        self.num_scans = num_scans
        self.future_impute_masking = future_impute_masking
        self.mask_window = mask_window

        # Grouping subjects
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        # Load SVF mapping: {subject -> {visit -> svf_path}}
        svf_df = pd.read_csv(svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            subj = row['Subject']
            visit = row['Visit']
            svf_path = row['SVF_Path']
            self.svf_map.setdefault(subj, {})[visit] = svf_path

        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        elif self.target_timepoint == "all":
            self.subject_combinations = []
            for subject in self.subjects:
                available_visits = set(self.df[self.df['Subject'] == subject]['Visit'].unique())
                valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]
                
                for target_tp in valid_timepoints:
                    if target_tp in available_visits:
                        self.subject_combinations.append((subject, target_tp))
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            self.subjects = list(self.subject_groups.index.levels[0])

        self.file_path_to_id = {}
        for i, row in self.df.iterrows():
            file_path = row['File_Path']
            subject_id = row['Subject']
            visit = row['Visit']
            uid = hash(file_path) % (10 ** 8)
            self.file_path_to_id[file_path] = uid

    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map

    def __len__(self):
        if self.target_timepoint == "all":
            return len(self.subject_combinations)
        return len(self.subjects)

    def load_scan(self, subject_id, visit='sc'):
        visit_data = self.df[(self.df['Subject'] == subject_id) & (self.df['Visit'] == visit)]
        if visit_data.empty:
            raise ValueError(f"No {visit} scan found for subject {subject_id}")
        file_path = visit_data.iloc[0]['File_Path']
        nii = nib.load(file_path)
        scan = nii.get_fdata()
        if scan.ndim == 3:
            scan = np.expand_dims(scan, axis=0)

        scan = self.test_transforms(scan)
        return scan
    
    def get_file_path_mapping(self):
        uid_to_file_path = {}
        for file_path, uid in self.file_path_to_id.items():
            if uid not in uid_to_file_path:
                uid_to_file_path[uid] = file_path
        return uid_to_file_path

    def _apply_original_masking(self, subject_id, target_timepoint, subject_data):
        mask = []
        svfs = []
        
        target_timepoint_index = self.timepoints.index(target_timepoint) - 1
        for t, timepoint in enumerate(self.timepoints):
            if timepoint == 'sc':
                continue

            if timepoint == target_timepoint:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))   
                continue

            if self.opt.use_future_data == False:
                if self.target_timepoint == "all":
                    if t > self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t > self.timepoints.index(target_timepoint):
                        continue

            if self.opt.use_past_data == False:
                if self.target_timepoint == "all":
                    if t < self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t < self.timepoints.index(target_timepoint):
                        continue
                
            visit_row = subject_data[subject_data['Visit'] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
            else:
                mask.append(1)
                try:
                    svf_path = self.svf_map[subject_id][timepoint]
                    svf = torch.load(svf_path).numpy()
                except KeyError:
                    svf = np.zeros((3, 128, 128, 128))
                svfs.append(svf)
                if svf.shape != (3, 128, 128, 128):
                    raise ValueError(f"SVF shape mismatch for subject {subject_id}, timepoint {timepoint}: {svf.shape}")
        
        return mask, svfs

    def _apply_robustness_masking(self, subject_id, target_timepoint, subject_data):
        mask = []
        svfs = []
        
        non_sc_timepoints = self.timepoints[1:]
        target_idx = non_sc_timepoints.index(target_timepoint)
        
        for timepoint in non_sc_timepoints:
            visit_row = subject_data[subject_data['Visit'] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
            else:
                mask.append(1)
                try:
                    svf_path = self.svf_map[subject_id][timepoint]
                    svf = torch.load(svf_path).numpy()
                except KeyError:
                    svf = np.zeros((3, 128, 128, 128))
                svfs.append(svf)
                if svf.shape != (3, 128, 128, 128):
                    raise ValueError(f"SVF shape mismatch for subject {subject_id}, timepoint {timepoint}: {svf.shape}")
        
        mask[target_idx] = 0
        svfs[target_idx] = np.zeros((3, 128, 128, 128))
        
        available_indices = [i for i in range(len(mask)) if mask[i] == 1]
        total_available = len(available_indices) + 1  # +1 for target
        
        if self.future_impute_masking and target_idx > 0:
            prev_idx = target_idx - 1
            if mask[prev_idx] == 1:
                mask[prev_idx] = 0
                svfs[prev_idx] = np.zeros((3, 128, 128, 128))
                if prev_idx in available_indices:
                    available_indices.remove(prev_idx)
        
        num_to_mask = max(0, total_available - self.num_scans - 1)  # -1 for target already masked
        
        if num_to_mask > 0 and len(available_indices) > 0:
            if self.mask_window:
                if num_to_mask <= len(available_indices):
                    window_positions = []
                    for start_pos in range(len(available_indices) - num_to_mask + 1):
                        window = available_indices[start_pos:start_pos + num_to_mask]
                        if any(abs(idx - target_idx) <= 1 for idx in window) or target_idx in range(min(window), max(window) + 1):
                            window_positions.append(window)
                    
                    if window_positions:
                        chosen_window = random.choice(window_positions)
                        for idx in chosen_window:
                            mask[idx] = 0
                            svfs[idx] = np.zeros((3, 128, 128, 128))
            else:
                indices_to_mask = random.sample(available_indices, min(num_to_mask, len(available_indices)))
                for idx in indices_to_mask:
                    mask[idx] = 0
                    svfs[idx] = np.zeros((3, 128, 128, 128))
        
        return mask, svfs

    def __getitem__(self, idx):
        if self.target_timepoint == "all":
            subject_id, target_timepoint = self.subject_combinations[idx]
        else:
            subject_id = self.subjects[idx]
            target_timepoint = self.subject_target_timepoints[subject_id]

        subject_data = self.subject_groups.loc[subject_id]
        
        latest_scan = self.load_scan(subject_id, visit='sc')
        target_scan = self.load_scan(subject_id, visit=target_timepoint)
        target_scan_path = subject_data[subject_data['Visit'] == target_timepoint].iloc[0]['File_Path']
        target_uid = self.file_path_to_id[target_scan_path]

        target_svf_path = self.svf_map[subject_id][target_timepoint]
        target_svf = torch.load(target_svf_path).numpy()

        use_robustness = (hasattr(self.opt, 'use_past_data') and hasattr(self.opt, 'use_future_data') and 
                         self.opt.use_past_data and self.opt.use_future_data)
        
        if use_robustness:
            mask, svfs = self._apply_robustness_masking(subject_id, target_timepoint, subject_data)
        else:
            mask, svfs = self._apply_original_masking(subject_id, target_timepoint, subject_data)
                
        stacked_svf = torch.tensor(np.stack(svfs)).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()
        mask = torch.tensor(mask, dtype=torch.float32)

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}
        

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map['sc']).unsqueeze(0)
        target_timepoint_index = torch.tensor(self.timepoints.index(target_timepoint) - 1).unsqueeze(0)
        target_svf = torch.tensor(target_svf).float()
        target_uid = torch.tensor(target_uid).unsqueeze(0).int()

        return {
            "stacked_svf": stacked_svf,
            "latest_scan": latest_scan,
            "target_scan": target_scan,
            "class_label": class_label,
            "reg_label": reg_label,
            "target_timepoint": target_timepoint_tensor,
            "latest_timepoint": latest_scan_timepoint,
            "mask": mask,
            "target_timepoint_index": target_timepoint_index,
            "target_svf": target_svf,
            "target_uid": target_uid
        }

class SVFUltraMaskedSequenceDataset(Dataset):
    def __init__(self, opt, csv_file, svf_file, fold, subset='train', aug_p=0.8,
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18',
                 log_missing_scans=False, use_val_dataset=False):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.opt = opt

        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        self.subset = subset
        self.aug_p = aug_p
        self.log_missing_scans = log_missing_scans

        # Grouping subjects
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        # Load SVF mapping: {subject -> {visit -> svf_path}}
        svf_df = pd.read_csv(svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            subj = row['Subject']
            visit = row['Visit']
            svf_path = row['SVF_Path']
            self.svf_map.setdefault(subj, {})[visit] = svf_path

        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        elif self.target_timepoint == "all":
            # For each subject, create multiple entries with different target timepoints
            self.subject_combinations = []
            for subject in self.subjects:
                # Get the actually available timepoints for this subject
                available_visits = set(self.df[self.df['Subject'] == subject]['Visit'].unique())
                valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]  # Skip 'sc'
                
                # Only include timepoints that have actual scans available
                for target_tp in valid_timepoints:
                    # Check if target timepoint scan actually exists
                    if target_tp in available_visits:
                        self.subject_combinations.append((subject, target_tp))
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            self.subjects = list(self.subject_groups.index.levels[0])

        # create a mapping between file path and id so we can return the id in the dataloader and reconvert it to filepath string
        self.file_path_to_id = {}
        for i, row in self.df.iterrows():
            file_path = row['File_Path']
            subject_id = row['Subject']
            visit = row['Visit']
            # generate an int uid
            uid = hash(file_path) % (10 ** 8)
            self.file_path_to_id[file_path] = uid

    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map

    def __len__(self):
        if self.target_timepoint == "all":
            return len(self.subject_combinations)
        return len(self.subjects)

    def load_scan(self, subject_id, visit='sc'):
        visit_data = self.df[(self.df['Subject'] == subject_id) & (self.df['Visit'] == visit)]
        if visit_data.empty:
            raise ValueError(f"No {visit} scan found for subject {subject_id}")
        file_path = visit_data.iloc[0]['File_Path']
        nii = nib.load(file_path)
        scan = nii.get_fdata()
        if scan.ndim == 3:
            scan = np.expand_dims(scan, axis=0)  # add channel dimension

        # apply transform
        scan = self.test_transforms(scan)
        return scan
    
    # def a function to get file_path mapping externally
    def get_file_path_mapping(self):
        uid_to_file_path = {}
        for file_path, uid in self.file_path_to_id.items():
            if uid not in uid_to_file_path:
                uid_to_file_path[uid] = file_path
        return uid_to_file_path

    def __getitem__(self, idx):
        if self.target_timepoint == "all":
            subject_id, target_timepoint = self.subject_combinations[idx]
        else:
            subject_id = self.subjects[idx]
            target_timepoint = self.subject_target_timepoints[subject_id]

        subject_data = self.subject_groups.loc[subject_id]
        
        mask = []

        # Load screening scan as latest_scan
        latest_scan = self.load_scan(subject_id, visit='sc')

        # Load target scan
        target_scan = self.load_scan(subject_id, visit=target_timepoint)
        target_scan_path = subject_data[subject_data['Visit'] == target_timepoint].iloc[0]['File_Path']
        target_uid = self.file_path_to_id[target_scan_path]

        # SVF stack for all timepoints except sc
        svfs = []
        # get the target SVF field
        target_svf_path = self.svf_map[subject_id][target_timepoint]
        target_svf = torch.load(target_svf_path).numpy()  # [3, H, W, D]

        target_timepoint_index = self.timepoints.index(target_timepoint) - 1
        for t, timepoint in enumerate(self.timepoints):
            if timepoint == 'sc':
                continue

            if timepoint == target_timepoint:
                # add mask
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))   
                continue

            # if not self.opt.use_future_data then skip scans after target_timepoint
            # if self.target_timepoint == "all" then add a mask instead of skipping
            if self.opt.use_future_data == False:
                if self.target_timepoint == "all":
                    if t > self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t > self.timepoints.index(target_timepoint):
                        continue

            if self.opt.use_past_data == False:
                if self.target_timepoint == "all":
                    if t < self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t < self.timepoints.index(target_timepoint):
                        continue
                
            visit_row = subject_data[subject_data['Visit'] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
            else:
                mask.append(1)
                try:
                    svf_path = self.svf_map[subject_id][timepoint]
                    svf = torch.load(svf_path).numpy()  # [3, H, W, D]
                except KeyError:
                    svf = np.zeros((3, 128, 128, 128))
                svfs.append(svf)
                if svf.shape != (3, 128, 128, 128):
                    raise ValueError(f"SVF shape mismatch for subject {subject_id}, timepoint {timepoint}: {svf.shape}")
                
        stacked_svf = torch.tensor(np.stack(svfs)).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()
        mask = torch.tensor(mask, dtype=torch.float32)

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map['sc']).unsqueeze(0)
        target_timepoint_index = torch.tensor(target_timepoint_index).unsqueeze(0)
        target_svf = torch.tensor(target_svf).float()
        target_uid = torch.tensor(target_uid).unsqueeze(0).int()

        return {
            "stacked_svf": stacked_svf,                       # [T-1, 3, H, W, D]
            "latest_scan": latest_scan,                       # [1, H, W, D]
            "target_scan": target_scan,                       # [1, H, W, D]
            "class_label": class_label,                       # [1]
            "reg_label": reg_label,                           # [1]
            "target_timepoint": target_timepoint_tensor,      # [1]
            "latest_timepoint": latest_scan_timepoint,        # [1]
            "mask": mask,                                     # [T]
            "target_timepoint_index": target_timepoint_index, # [1]
            "target_svf": target_svf,                         # [3, H, W, D]
            "target_uid": target_uid                          # [1]
        }

class SVFRandomMaskedSequenceDataset(Dataset):
    def __init__(self, opt, csv_file, svf_file, fold, subset='train', aug_p=0.8,
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18',
                 log_missing_scans=False, use_val_dataset=False):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.opt = opt

        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        self.subset = subset
        self.aug_p = aug_p
        self.log_missing_scans = log_missing_scans

        # Grouping subjects
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        # Load SVF mapping: {subject -> {visit -> svf_path}}
        svf_df = pd.read_csv(svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            subj = row['Subject']
            visit = row['Visit']
            svf_path = row['SVF_Path']
            self.svf_map.setdefault(subj, {})[visit] = svf_path

        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            self.subjects = list(self.subject_groups.index.levels[0])

    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map

    def __len__(self):
        return len(self.subjects)

    def load_scan(self, subject_id, visit='sc'):
        visit = self.df[(self.df['Subject'] == subject_id) & (self.df['Visit'] == visit)]
        if visit.empty:
            raise ValueError(f"No {visit} scan found for subject {subject_id}")
        file_path = visit.iloc[0]['File_Path']
        nii = nib.load(file_path)
        scan = nii.get_fdata()
        if scan.ndim == 3:
            scan = np.expand_dims(scan, axis=0)  # add channel dimension

        # apply transform
        scan = self.test_transforms(scan)
        return scan

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.subject_groups.loc[subject_id]
        target_timepoint = self.subject_target_timepoints[subject_id]

        mask = []

        # Load screening scan as latest_scan
        latest_scan = self.load_scan(subject_id, visit='sc')

        # Load target scan
        target_scan = self.load_scan(subject_id, visit=target_timepoint)

        # SVF stack for all timepoints except sc
        svfs = []
        for t, timepoint in enumerate(self.timepoints):
            if timepoint == 'sc':
                continue

            if timepoint == target_timepoint:
                # add mask
                mask.append(1)
                svfs.append(np.zeros((3, 128, 128, 128)))
                continue

            # if not self.opt.use_future_data then skip scans after target_timepoint
            if self.opt.use_future_data == False:
                if t > self.timepoints.index(target_timepoint):
                    continue

            if self.opt.use_past_data == False:
                if t < self.timepoints.index(target_timepoint):
                    continue
                
            visit_row = subject_data[subject_data['Visit'] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
            else:
                mask.append(1)
                try:
                    svf_path = self.svf_map[subject_id][timepoint]
                    svf = torch.load(svf_path).numpy()  # [3, H, W, D]
                except KeyError:
                    svf = np.zeros((3, 128, 128, 128))
                svfs.append(svf)
                if svf.shape != (3, 128, 128, 128):
                    raise ValueError(f"SVF shape mismatch for subject {subject_id}, timepoint {timepoint}: {svf.shape}")
                
        stacked_svf = torch.tensor(np.stack(svfs)).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()
        mask = torch.tensor(mask, dtype=torch.float32)

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map['sc']).unsqueeze(0)

        return {
            "stacked_svf": stacked_svf,                   # [T-1, 3, H, W, D]
            "latest_scan": latest_scan,                   # [1, H, W, D]
            "target_scan": target_scan,                   # [1, H, W, D]
            "class_label": class_label,                   # [1]
            "reg_label": reg_label,                       # [1]
            "target_timepoint": target_timepoint_tensor,  # [1]
            "latest_timepoint": latest_scan_timepoint,    # [1]
            "mask": mask                                   # [T]
        }



class ADNIRandomMaskedSequenceDataset(Dataset):
    def __init__(self, csv_file, fold, subset='train', aug_p=0.8, timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18', log_missing_scans=False, use_val_dataset=False):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])  
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')
        
        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])
        self.subset = subset
        self.aug_p = aug_p
        
        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        self.log_missing_scans = log_missing_scans
        
        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            # drop subjects missing the target_timepoint
            subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            self.subjects = list(self.subject_groups.index.levels[0])
    
    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:-1] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map
    
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.subject_groups.loc[subject_id]
        target_timepoint = self.subject_target_timepoints[subject_id]

        images = []
        mask = []

        for timepoint in self.timepoints:
            matched_scans = subject_data[subject_data['Visit'] == timepoint]

            if not matched_scans.empty:
                file_path = matched_scans.iloc[-1]['File_Path']
                nii = nib.load(file_path)
                img = nii.get_fdata()

                if img.ndim == 3:
                    img = np.expand_dims(img, axis=0)

                images.append(img)
                mask.append(1)
            else:
                images.append(np.zeros_like(images[0]) if images else np.zeros((1, 64, 64, 64)))
                mask.append(0)

        images = np.stack(images)
        mask = torch.tensor(mask, dtype=torch.float32)

        target_idx = self.timepoints.index(target_timepoint)

        previous_scans = images[:target_idx]
        latest_scan_idx = 0  # permanently fix to SC scan
        # latest_scan_idx = target_idx - 1
        # while latest_scan_idx >= 0 and mask[latest_scan_idx] == 0:
        #     latest_scan_idx -= 1

        latest_scan = images[latest_scan_idx]
        target_scan = images[target_idx]
        future_scans = images[target_idx + 1:] if target_idx + 1 < len(images) else np.zeros((0, *images[0].shape))

        if self.subset == 'train' and self.aug_p > 0:
            previous_scans = np.stack([self.monai_transforms(scan) for scan in previous_scans])
            latest_scan = self.monai_transforms(latest_scan)
            target_scan = self.monai_transforms(target_scan)
            future_scans = np.stack([self.monai_transforms(scan) for scan in future_scans]) if future_scans.size > 0 else future_scans
        else:
            previous_scans = np.stack([self.test_transforms(scan) for scan in previous_scans])
            latest_scan = self.test_transforms(latest_scan)
            target_scan = self.test_transforms(target_scan)
            future_scans = np.stack([self.test_transforms(scan) for scan in future_scans]) if future_scans.size > 0 else future_scans

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map[self.timepoints[latest_scan_idx]]).unsqueeze(0)

        mask_scan = torch.zeros_like(target_scan).unsqueeze(0)
        stacked_scan = np.concatenate([previous_scans, mask_scan, future_scans], axis=0)

        stacked_scan = torch.tensor(stacked_scan).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()

        return {
            "stacked_scan": stacked_scan,
            "latest_scan": latest_scan,
            "target_scan": target_scan,
            "class_label": class_label,
            "reg_label": reg_label,
            "target_timepoint": target_timepoint_tensor,
            "latest_timepoint": latest_scan_timepoint,
        }

class ADNIVxmTraining(Dataset):
    def __init__(self, csv_file, fold, subset='train', aug_p=0.8, timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18', log_missing_scans=False, use_val_dataset=False):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])  
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')
        
        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])
        self.subset = subset
        self.aug_p = aug_p
        
        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        self.log_missing_scans = log_missing_scans
        
        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            # drop subjects missing the target_timepoint
            # subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            # self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            # self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            # self.subjects = list(self.subject_groups.index.levels[0])
    
    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:-1] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map
    
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.subject_groups.loc[subject_id]
        target_timepoint = self.subject_target_timepoints[subject_id]

        images = []
        mask = []

        for timepoint in self.timepoints:
            matched_scans = subject_data[subject_data['Visit'] == timepoint]

            if not matched_scans.empty:
                file_path = matched_scans.iloc[-1]['File_Path']
                nii = nib.load(file_path)
                img = nii.get_fdata()

                if img.ndim == 3:
                    img = np.expand_dims(img, axis=0)

                images.append(img)
                mask.append(1)
            else:
                images.append(np.zeros_like(images[0]) if images else np.zeros((1, 64, 64, 64)))
                mask.append(0)

        images = np.stack(images)
        mask = torch.tensor(mask, dtype=torch.float32)

        target_idx = self.timepoints.index(target_timepoint)

        previous_scans = images[:target_idx]
        latest_scan_idx = 0 # fix to sc scan always

        latest_scan = images[latest_scan_idx]
        target_scan = images[target_idx]
        future_scans = images[target_idx + 1:] if target_idx + 1 < len(images) else np.zeros((0, *images[0].shape))

        if self.subset == 'train' and self.aug_p > 0:
            previous_scans = np.stack([self.monai_transforms(scan) for scan in previous_scans])
            latest_scan = self.monai_transforms(latest_scan)
            target_scan = self.monai_transforms(target_scan)
            future_scans = np.stack([self.monai_transforms(scan) for scan in future_scans]) if future_scans.size > 0 else future_scans
        else:
            previous_scans = np.stack([self.test_transforms(scan) for scan in previous_scans])
            latest_scan = self.test_transforms(latest_scan)
            target_scan = self.test_transforms(target_scan)
            future_scans = np.stack([self.test_transforms(scan) for scan in future_scans]) if future_scans.size > 0 else future_scans

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map[self.timepoints[latest_scan_idx]]).unsqueeze(0)

        mask_scan = torch.clone(target_scan).unsqueeze(0)
        stacked_scan = np.concatenate([previous_scans, mask_scan, future_scans], axis=0)

        stacked_scan = torch.tensor(stacked_scan).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()

        return {
            "stacked_scan": stacked_scan,
            "latest_scan": latest_scan,
            "target_scan": target_scan,
            "class_label": class_label,
            "reg_label": reg_label,
            "target_timepoint": target_timepoint_tensor,
            "latest_timepoint": latest_scan_timepoint,
            "mask": mask
        }


class ADNISequenceDataset(Dataset):
    """
    A dataset class that loads all scans of a given subject as a single data point.
    The scans are ordered by acquisition date to form a temporal sequence.
    Uses masking to handle missing timepoints and removes duplicate scans by keeping the latest scan for each visit.
    Suitable for use in sequence models such as transformers.
    """
    def __init__(self, csv_file, fold, subset='train', aug_p=0.8, timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18', log_missing_scans=False, use_val_dataset=False):
        """
        :param csv_file: Path to the CSV file containing the metadata.
        :param subset: Subset type ('train' or 'test') to determine which transforms to apply.
        :param aug_p: The augmentation probability.
        :param timepoints: Ordered list of expected timepoints for masking missing scans.
        """
        # Load the CSV into a pandas DataFrame
        self.df = pd.read_csv(csv_file)

        # Filter by fold
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)

        # if subset is 'train', then append the validation set to the training set
        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)
        
        # Convert acquisition date to datetime for sorting
        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])  
        
        # Remove duplicate scans, keeping only the latest scan per visit per subject
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')
        
        self.target_timepoint = target_timepoint
        
        # Drop subjects missing the target_timepoint
        subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
        self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)

        # Group by subject and sort by acquisition date
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        
        self.subjects = list(self.subject_groups.index.levels[0])  # Unique subjects
        self.subset = subset
        self.aug_p = aug_p
        self.timepoints = timepoints
        
        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            t.NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        self.log_missing_scans = log_missing_scans
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.subject_groups.loc[subject_id]

        images = []
        mask = []

        for timepoint in self.timepoints:
            matched_scans = subject_data[subject_data['Visit'] == timepoint]

            if not matched_scans.empty:
                file_path = matched_scans.iloc[-1]['File_Path']
                img = nib.load(file_path).get_fdata()
                if img.ndim == 3:
                    img = np.expand_dims(img, axis=0)
                images.append(img)
                mask.append(1)
            else:
                # check if target timepoint is missing
                if timepoint == self.target_timepoint:
                    raise ValueError(f"Missing scan for subject {subject_id} at timepoint {timepoint}")
                if self.log_missing_scans:
                    print(f"Missing scan for subject {subject_id} at timepoint {timepoint}")
                images.append(np.zeros_like(images[0]) if images else np.zeros((1, 64, 64, 64)))
                mask.append(0)

        images = np.stack(images)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Dynamically select based on target_timepoint
        target_idx = self.timepoints.index(self.target_timepoint)

        previous_scans = images[:target_idx]  # All scans before target_timepoint
        latest_scan = images[target_idx-1]      # Scan at just before target_timepoint
        target_scan = images[target_idx]        # Scan at target_timepoint
        future_scans = images[target_idx + 1:] if target_idx + 1 < len(images) else np.zeros((0, *images[0].shape))

        if self.subset == 'train' and self.aug_p > 0:
            previous_scans = np.stack([self.monai_transforms(scan) for scan in previous_scans])
            latest_scan = self.monai_transforms(latest_scan)
            target_scan = self.monai_transforms(target_scan)
            future_scans = np.stack([self.monai_transforms(scan) for scan in future_scans]) if future_scans.size > 0 else future_scans
        else:
            previous_scans = np.stack([self.test_transforms(scan) for scan in previous_scans])
            latest_scan = self.test_transforms(latest_scan)
            target_scan = self.test_transforms(target_scan)
            future_scans = np.stack([self.test_transforms(scan) for scan in future_scans]) if future_scans.size > 0 else future_scans

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)

        return previous_scans, latest_scan, target_scan, future_scans, class_label, reg_label





class ADNIDataset(Dataset):
    """
    A dataset class that reads a CSV file with image metadata and loads brain images.
    The CSV is expected to have columns including 'File_Path' and 'Age'.
    The file paths point to the NiFTI files, and the age is used as the label.
    """
    def __init__(self, csv_file, subset='train', aug_p=0.8):
        """
        :param csv_file: Path to the CSV file containing the metadata.
        :param subset: Subset type ('train' or 'test') to determine which transforms to apply.
        :param aug_p: The augmentation probability.
        """
        # Load the CSV into a pandas DataFrame
        self.df = pd.read_csv(csv_file)
        # Extract file paths and age labels
        self.filepaths = self.df['File_Path'].tolist()

        # Classification labels
        self.class_labels = pd.Categorical(self.df['Sex']).codes.astype(float).tolist()
        # Regression labels
        self.reg_labels = self.df['Age'].astype(float).tolist()

        self.subset = subset
        self.aug_p = aug_p

        # Define MONAI transforms for training and testing
        self.test_transforms = t.Compose([t.NormalizeIntensity()])
        self.monai_transforms = t.Compose([
            ToTensor(),
            t.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                             prob=aug_p, rotate_range=(0.18, 0.18, 0.18),
                             translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                             spatial_size=None, padding_mode="border"),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Get the file path for the current index
        file_path = self.filepaths[idx]
        # Load the image using nibabel
        img = nib.load(file_path)
        image = img.get_fdata()

        # If the image is 3D, add a channel dimension (e.g., for single-channel MRI data)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
    
        # Apply transforms based on the subset
        if self.subset == 'train' and self.aug_p > 0:
            image = self.monai_transforms(image)
        else:
            image = self.test_transforms(image)

        class_label = torch.tensor(self.class_labels[idx]).unsqueeze(0)
        reg_label = torch.tensor(self.reg_labels[idx]).unsqueeze(0)

        return image, class_label, reg_label

class ExampleDataset(Dataset):
    '''
    This is an example Dataset class that you should adapt to use your own data
    here we produce just random tensors, to illustrate the augmentation pipeline and how it would fit in
    the train_model.py code. For the paper we used UK Biobank data which is freely avaible, but not shareable
    '''

    def __init__(self, subset, aug_p=0.8):
        '''
        Recommended to add oversampling to ensure the generator learns the task properly
        :param aug_p: The augmentation probability. Due to the difficulty of the task and the lack of brain imaging
        available it is recommended that this be set to a fairly high value.
        '''

        self.aug_p = aug_p
        self.subset = subset
        self.test_transforms = t.Compose([t.NormalizeIntensity()])
        self.monai_transforms = [ToTensor(),
                                 t.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                                                 prob=aug_p, rotate_range=(0.18, 0.18, 0.18),
                                                 translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                                                 spatial_size=None, padding_mode="border", as_tensor_output=False),
                                 t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
                                 t.RandAdjustContrast(prob=aug_p),
                                 t.RandGaussianNoise(prob=aug_p),
                                 t.NormalizeIntensity()]
        self.monai_transforms = t.Compose(self.monai_transforms)

        # Where your brains would go
        self.brain_volumes = np.random.random((1000, 64, 64, 64))
        # Classification labels
        self.class_labels = np.random.random((1000, 1))
        # Regression labels
        self.reg_labels = np.random.random((1000, 1))

    def __len__(self):
        return len(self.brain_volumes)

    def __getitem__(self, index):
        image = np.expand_dims(self.brain_volumes[index], axis=0)

        if self.subset == 'train' and self.aug_p > 0:
            image = self.monai_transforms(image)
        else:
            image = self.test_transforms(image)

        return image, self.class_labels[index], self.reg_labels[index]


class MultiEpochsDataLoader(MonaiDataLoader):
    '''
    Override the default dataloader so that it doesn't spawn loads of processes at the start of every epoch
    This saves a huge amount of time for 3D
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Nicer for tqdm
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.tracker = []

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def track(self):
        """
        @rtype: object
        """
        self.tracker.append(self.avg)

    def save(self, fn):
        np.save(fn, np.array(self.tracker))
