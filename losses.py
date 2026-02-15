# =============================================================================
# File: losses.py
# Purpose: Losses and metric utilities (alignment, Jacobian, similarity, EPE/angular, integration helpers).
# Prerequisites: Imported by all major train/test/eval scripts.
# Usually Run Next: Provides objective terms and metrics for both checkpoint selection and report generation.
# Model Components Used: No model definitions; shared metric/loss backend for active SYNBADD pipelines.
# =============================================================================

import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch import nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from pytorch_msssim import ssim as ssim_loss
from pytorch_msssim import ms_ssim
import pandas as pd
import os

class SafeUpsample(nn.Module):
    """
    A safe upsampling module that handles large 3D volumes by processing them in chunks if needed.
    """
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        batch_size, channels, d, h, w = x.shape
        
        # Check if regular upsampling would exceed int32 limit
        output_size = (d * self.scale_factor, h * self.scale_factor, w * self.scale_factor)
        total_elements = batch_size * channels * output_size[0] * output_size[1] * output_size[2]
        
        if total_elements < 2**31:  # If under int32 limit, use regular upsampling
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        
        # Otherwise, process in chunks along the depth dimension
        chunk_size = max(1, d // 4)  # Process 1/4 of depth at a time
        output_chunks = []
        
        for i in range(0, d, chunk_size):
            end_idx = min(i + chunk_size, d)
            chunk = x[:, :, i:end_idx, :, :]
            upsampled_chunk = F.interpolate(chunk, 
                                          scale_factor=self.scale_factor, 
                                          mode=self.mode)
            output_chunks.append(upsampled_chunk)
            
        return torch.cat(output_chunks, dim=2)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', get_unnormed=False):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        self.register_buffer('grid', grid)
        self.get_unnormed = get_unnormed

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        unnormed_locs = torch.clone(new_locs)

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # For visualisation purposes
        if self.get_unnormed:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), unnormed_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

# Global cache
_cached_atlas = None
_cached_label_map = None

def channelwise_epe_loss(predicted_field, target_field):
    """
    Compute channelwise End-Point Error (EPE) loss for 3D deformation fields.
    
    Args:
        predicted_field (torch.Tensor): Predicted deformation field of shape [B, 3, H, W, D]
        target_field (torch.Tensor): Target deformation field of shape [B, 3, H, W, D]
    
    Returns:
        tuple: (total_epe_loss, x_epe_loss, y_epe_loss, z_epe_loss)
            - total_epe_loss: Average EPE loss across all channels
            - x_epe_loss: EPE loss for x channel
            - y_epe_loss: EPE loss for y channel
            - z_epe_loss: EPE loss for z channel
    """
    # Ensure both tensors are on the same device
    device = predicted_field.device
    
    # Compute L2 error for each channel
    diff = predicted_field - target_field
    
    # Compute channel-wise EPE (Euclidean distance for each channel)
    x_epe = torch.mean(torch.abs(diff[:, 0, ...]))
    y_epe = torch.mean(torch.abs(diff[:, 1, ...]))
    z_epe = torch.mean(torch.abs(diff[:, 2, ...]))
    
    # Total EPE is the average across channels
    total_epe = (x_epe + y_epe + z_epe) / 3.0
    
    return total_epe, x_epe, y_epe, z_epe

def angular_error_loss(predicted_field, target_field, eps=1e-6):
    """
    Compute angular error between predicted and target 3D deformation fields.
    
    Args:
        predicted_field (torch.Tensor): Predicted deformation field of shape [B, 3, H, W, D]
        target_field (torch.Tensor): Target deformation field of shape [B, 3, H, W, D]
        eps (float): Small constant to avoid division by zero
    
    Returns:
        torch.Tensor: Mean angular error in radians
    """
    import torch
    
    # Reshape tensors to [B, 3, -1] to treat each voxel as a 3D vector
    batch_size = predicted_field.shape[0]
    pred_flat = predicted_field.view(batch_size, 3, -1)
    target_flat = target_field.view(batch_size, 3, -1)
    
    # Compute vector magnitudes
    pred_mag = torch.sqrt(torch.sum(pred_flat**2, dim=1) + eps)  # [B, -1]
    target_mag = torch.sqrt(torch.sum(target_flat**2, dim=1) + eps)  # [B, -1]
    
    # Compute dot product
    dot_product = torch.sum(pred_flat * target_flat, dim=1)  # [B, -1]
    
    # Compute cosine similarity
    cos_sim = dot_product / (pred_mag * target_mag)
    
    # Clamp to valid range to avoid numerical issues
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    
    # Compute angular error in radians
    angular_error = torch.acos(cos_sim)  # [B, -1]
    
    # Return mean angular error
    return torch.mean(angular_error)

def ssim_only_loss(y_pred, y_true):
    return 1 - ssim_loss(y_pred, y_true, data_range=1.0)

def create_voxel_identity_grid(size, device):
    """
    Create a voxel-space identity grid of shape (1, 3, H, W, D)
    """
    H, W, D = size
    vectors = [torch.arange(0, s, device=device) for s in (H, W, D)]
    grid = torch.stack(torch.meshgrid(*vectors, indexing="ij"))  # (3, H, W, D)
    return grid.unsqueeze(0).float()  # (1, 3, H, W, D)

def integrate_svf(svf, int_steps=10, int_downsize=1):
    """
    Integrates a stationary velocity field (SVF) to produce a diffeomorphic deformation field.

    Args:
        svf (torch.Tensor): Tensor of shape (1, 3, H, W, D) — the stationary velocity field.
        int_steps (int): Number of scaling and squaring steps.
        int_downsize (int): Downsampling factor before integration (speeds up computation).

    Returns:
        torch.Tensor: Integrated deformation field of shape (1, 3, H, W, D).
    """
    assert svf.dim() == 5 and svf.shape[1] == 3, "SVF must have shape (B, 3, H, W, D)"
    device = svf.device
    B, C, H, W, D = svf.shape

    # Downsize if required
    if int_downsize > 1:
        resize = ResizeTransform(int_downsize, ndims=3).to(device)
        fullsize = ResizeTransform(1 / int_downsize, ndims=3).to(device)
        svf_down = resize(svf)
    else:
        svf_down = svf
        fullsize = lambda x: x  # Identity

    # Integrate
    integrator = VecInt(inshape=svf_down.shape[2:], nsteps=int_steps).to(device)
    deformation_field = integrator(svf_down)

    # Resize to original resolution if needed
    deformation_field = fullsize(deformation_field)

    return deformation_field

def jacobian_determinant_torch(flow):
    """
    Compute the Jacobian determinant of a 3D displacement field.
    flow: tensor (B, 3, H, W, D)
    returns: tensor (B, H-1, W-1, D-1)
    """
    # integrate flow to get mapping
    flow = integrate_svf(flow)  # [B, 3, H, W, D]

    B, C, H, W, D = flow.shape

    # Identity grid
    grid = create_voxel_identity_grid((H, W, D), flow.device)
    mapping = flow + grid  # φ(x)

    # Compute partial derivatives (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
    dfx = mapping[:, :, :, 1:, :] - mapping[:, :, :, :-1, :]
    dfy = mapping[:, :, 1:, :, :] - mapping[:, :, :-1, :, :]
    dfz = mapping[:, :, :, :, 1:] - mapping[:, :, :, :, :-1]

    # Crop to common shape (B, 3, H-1, W-1, D-1)
    dfx = dfx[:, :, :-1, :, :-1]  # remove last H and D
    dfy = dfy[:, :, :, :-1, :-1]  # remove last W and D
    dfz = dfz[:, :, :-1, :-1, :]  # remove last H and W

    # Stack: (B, 3, H-1, W-1, D-1, 3)
    J = torch.stack([dfy, dfx, dfz], dim=-1)
    J = J.permute(0, 2, 3, 4, 1, 5)  # (B, H-1, W-1, D-1, 3, 3)

    # Compute determinant of Jacobian matrix at each voxel
    jac_det = (
        J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1]) -
        J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0]) +
        J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0])
    )
    return jac_det


def gradient_penalty(y, x):
    """
    Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
    I don't use this but leaving here because a lot of GANs do use it. Worse results than R1
    For explanation see paper
    """
    weight = torch.ones(y.size()).to(y.device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def gradient_loss(flow):
    # Calculate spatial gradients of the flow field
    dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
    
    # Average
    return torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

def smooth_loss_l2(y_pred, penalty='l2', loss_mult=1):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    if loss_mult is not None:
        grad *= loss_mult
    return grad

def multi_scale_loss(y_synthetic, target_scan):
    loss = F.mse_loss(y_synthetic, target_scan)  # Full resolution
    loss += F.mse_loss(F.avg_pool3d(y_synthetic, 2), F.avg_pool3d(target_scan, 2))  # Downsampled
    loss += F.mse_loss(F.avg_pool3d(y_synthetic, 4), F.avg_pool3d(target_scan, 4))  # More downsampling

    return loss

def compute_metrics(target_img, generated_img, synthetic_field, lddmmloss, device="cuda"):
    target_img = target_img.to(device)
    generated_img = generated_img.to(device)

    batch_size = target_img.shape[0]

    # Normalize each image to mean=0, std=1 using instance norm
    norm_target_img = target_img
    norm_generated_img = generated_img

    # Rescale to [0,1] range for compatibility with metrics
    norm_target_img = (norm_target_img - norm_target_img.min()) / (norm_target_img.max() - norm_target_img.min() + 1e-8)
    norm_generated_img = (norm_generated_img - norm_generated_img.min()) / (norm_generated_img.max() - norm_generated_img.min() + 1e-8)

    # Compute intensity similarity metrics (across entire batch)
    mse_loss = F.mse_loss(norm_generated_img, norm_target_img).item()

    # Calculate NRMSE per image using individual std then average
    nrmse_values = []
    for index in range(batch_size):
        # Get single image from batch
        target_img_single = target_img[index]
        norm_target_single = norm_target_img[index]
        norm_generated_single = norm_generated_img[index]

        # Get original std of this specific image (before normalization)
        original_target_std = torch.std(target_img_single) + 1e-8

        # Calculate RMSE for this image
        img_rmse = torch.sqrt(torch.mean((norm_generated_single - norm_target_single) ** 2))

        # Normalize by this image's std
        img_nrmse = (img_rmse / original_target_std).item()
        nrmse_values.append(img_nrmse)

    # Average the per-image NRMSE values
    nrmse_loss = sum(nrmse_values) / len(nrmse_values)

    psnr_value = 0
    ssim_value = 0
    mssim_value = 0  # Initialize MS-SSIM value
    
    # Compute PSNR, SSIM & MS-SSIM across all batch elements
    for index in range(batch_size):
        # Get single image from batch
        target_img_single = target_img[index].cpu().detach().numpy()
        generated_img_single = generated_img[index].cpu().detach().numpy()

        # remove channel
        target_img_single = target_img_single[0]
        generated_img_single = generated_img_single[0]

        # choose image where it is > 0
        target_img_single = np.where(target_img_single > 0, target_img_single, 0)
        generated_img_single = np.where(generated_img_single > 0, generated_img_single, 0)
        
        # normalize
        target_img_single = (target_img_single - target_img_single.min()) / (target_img_single.max() - target_img_single.min()) * 255
        generated_img_single = (generated_img_single - generated_img_single.min()) / (generated_img_single.max() - generated_img_single.min()) * 255

        # convert to uint8
        target_img_single = target_img_single.astype(np.uint8)
        generated_img_single = generated_img_single.astype(np.uint8)

        # Calculate PSNR and SSIM
        psnr_value_t = psnr(generated_img_single, target_img_single)
        ssim_value_t = ssim(generated_img_single, target_img_single, multichannel=False)

        # Add to cumulative values
        psnr_value += psnr_value_t
        ssim_value += ssim_value_t
        
        # Calculate MS-SSIM using pytorch_msssim
        # Convert back to torch tensors and reshape for ms_ssim (needs [B,C,H,W] format)
        target_tensor = torch.from_numpy(target_img_single).float().unsqueeze(0).unsqueeze(0).to(device)
        generated_tensor = torch.from_numpy(generated_img_single).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Normalize tensors to [0,1] for ms_ssim
        target_tensor = target_tensor / 255.0
        generated_tensor = generated_tensor / 255.0
        
        # Calculate MS-SSIM (returns a tensor)
        # mssim_value_t = ms_ssim(
        #     generated_tensor, 
        #     target_tensor, 
        #     data_range=1.0,
        #     win_size=7,  # Smaller window size (default is 11)
        # ).item()
        # mssim_value += mssim_value_t
        mssim_value = 0

    # Average the per-image PSNR, SSIM and MS-SSIM values
    psnr_value /= batch_size
    ssim_value /= batch_size
    mssim_value /= batch_size  # Average MS-SSIM across batch
        
    jac_det = jacobian_determinant_torch(synthetic_field)  # [B, H-1, W-1, D-1]
    mean_jacobian = torch.mean(jac_det).item()
    std_jacobian = torch.std(jac_det).item()

    return mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, jac_det, nrmse_loss, mssim_value


class VxmLDDMMLoss(nn.Module):
    """
    Differentiable LDDMM-style loss using Voxelmorph.
    Uses a pretrained VxmDense model to compute flow fields and warped outputs.
    """
    def __init__(self, vxm_model, weight_jacobian=0.01, weight_intensity=1.0):
        super(VxmLDDMMLoss, self).__init__()
        self.vxm = vxm_model
        self.weight_jacobian = weight_jacobian
        self.weight_intensity = weight_intensity

    def multiscale_mse(self, x, y):
        """
        Multi-scale mean squared error.
        Compares x and y at 3 scales: full, 2x downsampled, and 4x downsampled.
        """
        loss = F.mse_loss(x, y)
        loss += F.mse_loss(F.avg_pool3d(x, 2), F.avg_pool3d(y, 2))
        loss += F.mse_loss(F.avg_pool3d(x, 4), F.avg_pool3d(y, 4))
        return loss / 3.0  # average across scales

    def forward(self, target_img, generated_img):
        """
        Parameters:
        - target_img: [B, 1, H, W, D] — fixed image (target)
        - generated_img: [B, 1, H, W, D] — moving image (generated output)

        Returns:
        - mean LDDMM-style loss over batch
        """
        # Compute deformation flow and warped image WITHOUT torch.no_grad()
        warped_img, flow_field = self.vxm(generated_img, target_img) # [B, 1, H, W, D], [B, 3, H, W, D]

        # Intensity similarity loss (MSE)
        intensity_loss = self.multiscale_mse(warped_img, target_img)

        # Jacobian regularization
        jacobian = jacobian_determinant_torch(flow_field)  # [B, H, W, D]
        jacobian_loss = torch.mean(torch.abs(jacobian - 1.0))

        # Combined loss
        total_loss = self.weight_intensity * intensity_loss + self.weight_jacobian * jacobian_loss
        return total_loss