"""Similarity losses and metrics (SSIM, PSNR, MSE, NRMSE)."""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as pytorch_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def ssim_loss(y_pred, y_true):
    """1 - SSIM loss for training."""
    return 1 - pytorch_ssim(y_pred, y_true, data_range=1.0)


def multi_scale_loss(y_synthetic, target_scan):
    """MSE at 3 scales: full, 2x downsampled, 4x downsampled."""
    loss = F.mse_loss(y_synthetic, target_scan)
    loss += F.mse_loss(F.avg_pool3d(y_synthetic, 2), F.avg_pool3d(target_scan, 2))
    loss += F.mse_loss(F.avg_pool3d(y_synthetic, 4), F.avg_pool3d(target_scan, 4))
    return loss


def compute_metrics(target_img, generated_img, synthetic_field, lddmmloss,
                    device="cuda"):
    """Compute full metric suite for validation/testing.

    Returns:
        tuple: (mse, psnr, ssim, mean_jac, std_jac, jac_det, nrmse, mssim)
    """
    from synbadd.losses.deformation import jacobian_determinant_torch

    target_img = target_img.to(device)
    generated_img = generated_img.to(device)
    batch_size = target_img.shape[0]

    # Normalize to [0, 1]
    norm_target = target_img
    norm_generated = generated_img
    norm_target = (norm_target - norm_target.min()) / (norm_target.max() - norm_target.min() + 1e-8)
    norm_generated = (norm_generated - norm_generated.min()) / (norm_generated.max() - norm_generated.min() + 1e-8)

    mse_value = F.mse_loss(norm_generated, norm_target).item()

    # Per-image NRMSE
    nrmse_values = []
    for i in range(batch_size):
        original_std = torch.std(target_img[i]) + 1e-8
        img_rmse = torch.sqrt(torch.mean((norm_generated[i] - norm_target[i]) ** 2))
        nrmse_values.append((img_rmse / original_std).item())
    nrmse_value = sum(nrmse_values) / len(nrmse_values)

    # Per-image PSNR and SSIM
    psnr_value = 0.0
    ssim_value = 0.0

    for i in range(batch_size):
        target_np = target_img[i, 0].cpu().detach().numpy()
        gen_np = generated_img[i, 0].cpu().detach().numpy()

        target_np = np.where(target_np > 0, target_np, 0)
        gen_np = np.where(gen_np > 0, gen_np, 0)

        target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8) * 255
        gen_np = (gen_np - gen_np.min()) / (gen_np.max() - gen_np.min() + 1e-8) * 255

        target_np = target_np.astype(np.uint8)
        gen_np = gen_np.astype(np.uint8)

        psnr_value += psnr(gen_np, target_np)
        ssim_value += ssim(gen_np, target_np, multichannel=False)

    psnr_value /= batch_size
    ssim_value /= batch_size
    mssim_value = 0.0  # MS-SSIM disabled in original

    # Jacobian statistics
    jac_det = jacobian_determinant_torch(synthetic_field)
    mean_jac = torch.mean(jac_det).item()
    std_jac = torch.std(jac_det).item()

    return mse_value, psnr_value, ssim_value, mean_jac, std_jac, jac_det, nrmse_value, mssim_value
