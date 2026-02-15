# =============================================================================
# File: scripts/compute_svf_stats.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import torch
from tqdm import tqdm
import numpy as np

SVF_DIR = "/scratch/anirudhkaushik/svf_outputs_backup"

def compute_svf_stats(svf_dir):
    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []
    max_abs_vals = []

    file_list = [f for f in os.listdir(svf_dir) if f.endswith(".pt")]
    print(f"Found {len(file_list)} SVF files.")

    for file in tqdm(file_list, desc="Processing SVFs"):
        svf_path = os.path.join(svf_dir, file)
        svf = torch.load(svf_path)  # Expected shape: [3, H, W, D]
        svf = svf.float()

        min_vals.append(svf.min().item())
        max_vals.append(svf.max().item())
        mean_vals.append(svf.mean().item())
        std_vals.append(svf.std().item())
        max_abs_vals.append(svf.abs().max().item())

    global_stats = {
        "global_min": float(np.min(min_vals)),
        "global_max": float(np.max(max_vals)),
        "global_mean": float(np.mean(mean_vals)),
        "global_std": float(np.mean(std_vals)),  # Mean of per-SVF stds
        "global_max_abs": float(np.max(max_abs_vals))
    }

    print("\n📊 SVF Dataset Statistics:")
    for k, v in global_stats.items():
        print(f"{k}: {v:.6f}")

    return global_stats

if __name__ == "__main__":
    stats = compute_svf_stats(SVF_DIR)
