"""Shared evaluation metrics: Dice, volume, ROI extraction, NIfTI loading."""

import os

import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_nii(path, target_shape=(128, 128, 128)):
    """Load a NIfTI file, squeeze if 4D, pad/center to target_shape, normalize to [0,1]."""
    nii = nib.load(path)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data.squeeze(0)

    # Pad or crop to target_shape
    if data.shape != target_shape:
        padded = np.zeros(target_shape, dtype=data.dtype)
        slices_src = []
        slices_dst = []
        for i in range(3):
            if data.shape[i] <= target_shape[i]:
                offset = (target_shape[i] - data.shape[i]) // 2
                slices_src.append(slice(0, data.shape[i]))
                slices_dst.append(slice(offset, offset + data.shape[i]))
            else:
                offset = (data.shape[i] - target_shape[i]) // 2
                slices_src.append(slice(offset, offset + target_shape[i]))
                slices_dst.append(slice(0, target_shape[i]))
        padded[tuple(slices_dst)] = data[tuple(slices_src)]
        data = padded

    # Min-max normalize to [0, 1]
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 1e-8:
        data = (data - dmin) / (dmax - dmin)
    data = np.clip(data, 0, 1)
    return data


def dice_score(pred_mask, target_mask):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(pred_mask * target_mask)
    return (2.0 * intersection) / (np.sum(pred_mask) + np.sum(target_mask) + 1e-6)


def extract_roi(seg_data, region_ids):
    """Extract binary mask for a set of region IDs from a segmentation volume."""
    mask = np.zeros_like(seg_data, dtype=np.float32)
    for rid in region_ids:
        mask[seg_data == rid] = 1.0
    return mask


def calculate_volume_mm3(mask, voxel_size=(2, 2, 2)):
    """Calculate volume in mm^3 from a binary mask."""
    voxel_vol = np.prod(voxel_size)
    return float(np.sum(mask) * voxel_vol)


def compute_ssim_psnr(target, generated):
    """Compute SSIM and PSNR between two 3D volumes (both in [0,1] range).

    Args:
        target: 3D numpy array.
        generated: 3D numpy array.

    Returns:
        (ssim_value, psnr_value)
    """
    # Convert to uint8 for standard metrics
    target_uint8 = (target * 255).astype(np.uint8)
    gen_uint8 = (generated * 255).astype(np.uint8)

    ssim_val = ssim(gen_uint8, target_uint8, data_range=255)
    psnr_val = psnr(target_uint8, gen_uint8)
    return ssim_val, psnr_val


def find_seg_file(seg_dir, image_id):
    """Find segmentation file matching an image ID in a directory.

    Looks for files containing the image_id and ending with the
    standard FreeSurfer atlas naming convention.
    """
    for fname in os.listdir(seg_dir):
        if image_id in fname and fname.endswith(".nii.gz"):
            return os.path.join(seg_dir, fname)
    return None
