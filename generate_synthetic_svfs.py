# =============================================================================
# File: generate_synthetic_svfs.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import random
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

# --------- Config ---------
NUM_AUG_PER_SAMPLE = 10  # Number of augmentations per original sample
SVF_FILE = "/scratch/anirudhkaushik/svf_metadata.csv"  # must have columns: Subject, Visit, SVF_Path
SAVE_DIR = "/scratch/anirudhkaushik/synthetic_deformations"
SVF_SHAPE = (3, 128, 128, 128)  # expected shape for SVFs

# Augmentation hyperparameters
ELASTIC_SIGMA_RANGE = (3.0, 6.0)
SCALE_RANGE = (1.2, 2.0)
AFFINE_RANGE = 0.05
ELASTIC_STRENGTH = 0.05
ROTATION_DEG = 10


# --------- Utils ---------
def apply_rotation_to_svf(svf, rot_matrix):
    c, h, w, d = svf.shape
    svf_flat = svf.reshape(3, -1)
    svf_rotated_flat = rot_matrix @ svf_flat
    return svf_rotated_flat.reshape(3, h, w, d)


def generate_augmented_svf(base_svf):
    svf_aug = base_svf.copy()

    # 1. Scaling
    scale = random.uniform(*SCALE_RANGE)
    svf_aug *= scale

    # 2. Elastic noise
    sigma = random.uniform(*ELASTIC_SIGMA_RANGE)
    elastic_noise = np.random.randn(*svf_aug.shape)
    elastic_noise = np.stack([
        gaussian_filter(elastic_noise[i], sigma=sigma) for i in range(3)
    ])
    svf_aug += ELASTIC_STRENGTH * elastic_noise

    # 3. Affine ramp
    affine_field = np.zeros_like(svf_aug)
    spatial_shape = svf_aug.shape[1:]
    for i in range(3):
        ramp = np.linspace(-1, 1, spatial_shape[i]) * random.uniform(-AFFINE_RANGE, AFFINE_RANGE)
        ramp_shape = [1, 1, 1]
        ramp_shape[i] = spatial_shape[i]
        ramp = ramp.reshape(ramp_shape)
        ramp_field = np.broadcast_to(ramp, spatial_shape)
        affine_field[i] = ramp_field
    svf_aug += affine_field

    # 4. Physically correct vector rotation
    angles = [random.uniform(-ROTATION_DEG, ROTATION_DEG) for _ in range(3)]
    rot_matrix = R.from_euler('xyz', angles, degrees=True).as_matrix()
    svf_aug = apply_rotation_to_svf(svf_aug, rot_matrix)

    return torch.tensor(svf_aug, dtype=torch.float32)


# --------- Main ---------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    svf_df = pd.read_csv(SVF_FILE)
    if not {'Subject', 'Visit', 'SVF_Path'}.issubset(svf_df.columns):
        raise ValueError("SVF file must have columns: Subject, Visit, SVF_Path")

    available_paths = svf_df['SVF_Path'].tolist()
    print(f"Loaded {len(available_paths)} real SVF paths.")

    total_augmentations = 0
    for real_path in available_paths:
        try:
            base_svf = torch.load(real_path).numpy()
            if base_svf.shape != SVF_SHAPE:
                print(f"Skipping malformed SVF: {real_path}")
                continue
        except Exception as e:
            print(f"Failed to load {real_path}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(real_path))[0]
        
        for aug_idx in range(NUM_AUG_PER_SAMPLE):
            synthetic_svf = generate_augmented_svf(base_svf)
            save_path = os.path.join(SAVE_DIR, f"{base_name}_aug{aug_idx}.pt")
            torch.save(synthetic_svf, save_path)
            total_augmentations += 1
            
            if total_augmentations % 50 == 0:
                print(f"[{total_augmentations}] Saved {save_path}")

    print(f"✅ Done! {total_augmentations} augmented SVFs saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()