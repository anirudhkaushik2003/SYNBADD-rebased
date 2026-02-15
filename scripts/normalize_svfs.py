# =============================================================================
# File: scripts/normalize_svfs.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import torch
from tqdm import tqdm

# --- CONFIG ---
SVF_DIR = "/scratch/anirudhkaushik/svf_outputs"
CLIP_MIN, CLIP_MAX = -10.0, 10.0  # physical displacement range (in mm)

def normalize_svf(svf, clip_min, clip_max):
    # Clip values to avoid outlier scaling
    svf = torch.clamp(svf, min=clip_min, max=clip_max)
    # Scale to [-1, 1]
    svf = 2 * ((svf - clip_min) / (clip_max - clip_min)) - 1
    return svf

def normalize_svfs_in_place(svf_dir, clip_min, clip_max):
    file_list = [f for f in os.listdir(svf_dir) if f.endswith(".pt")]
    print(f"Normalizing {len(file_list)} SVFs in: {svf_dir}")

    for file in tqdm(file_list, desc="Normalizing SVFs"):
        path = os.path.join(svf_dir, file)
        svf = torch.load(path)
        svf = svf.float()

        svf_norm = normalize_svf(svf, clip_min, clip_max)
        torch.save(svf_norm, path)  # Overwrite original file

    print("\n✅ Normalization complete and files updated in-place.")

if __name__ == "__main__":
    normalize_svfs_in_place(SVF_DIR, CLIP_MIN, CLIP_MAX)
