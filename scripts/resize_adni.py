# =============================================================================
# File: scripts/resize_adni.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import nibabel as nib
import numpy as np
import scipy.ndimage
from tqdm import tqdm

# Define input and output directories
INPUT_DIR = "/scratch/anirudhkaushik/Padded_Transposed_ADNI"
OUTPUT_DIR = "/scratch/anirudhkaushik/Resized_ADNI"
LOG_FILE = os.path.join(OUTPUT_DIR, "resize_log.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target shape
TARGET_SHAPE = (128, 128, 128)

def resize_image(input_path, output_path, log_file):
    """Resize a NIfTI image to the target shape."""
    try:
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Compute the zoom factor for resizing
        zoom_factors = np.array(TARGET_SHAPE) / np.array(data.shape)
        resized_data = scipy.ndimage.zoom(data, zoom_factors, order=3)  # Use cubic interpolation
        
        # Save the resized image
        new_img = nib.Nifti1Image(resized_data, affine=img.affine)
        nib.save(new_img, output_path)
        
        with open(log_file, "a") as log:
            log.write(f"Successfully resized: {input_path} -> {output_path}\n")
    except Exception as e:
        with open(log_file, "a") as log:
            log.write(f"Error resizing {input_path}: {str(e)}\n")

# Collect all NIfTI files
nii_files = [os.path.join(root, file) for root, _, files in os.walk(INPUT_DIR) for file in files if file.endswith(".nii") or file.endswith(".nii.gz")]

# Process all NIfTI files with a progress bar
with open(LOG_FILE, "w") as log:
    log.write("Starting resizing process...\n")

for input_path in tqdm(nii_files, desc="Resizing NIfTI Files", unit="file", ncols=80):
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
    resize_image(input_path, output_path, LOG_FILE)

with open(LOG_FILE, "a") as log:
    log.write("Resizing process completed.\n")

print(f"\nAll NIfTI files resized to {TARGET_SHAPE}. Log saved to", LOG_FILE)