"""Deformation field losses: Jacobian, EPE, angular error, SVF integration."""

import torch

from synbadd.models.spatial_ops import ResizeTransform, VecInt


def create_voxel_identity_grid(size, device):
    """Create a voxel-space identity grid of shape (1, 3, H, W, D)."""
    H, W, D = size
    vectors = [torch.arange(0, s, device=device) for s in (H, W, D)]
    grid = torch.stack(torch.meshgrid(*vectors, indexing="ij"))
    return grid.unsqueeze(0).float()


def integrate_svf(svf, int_steps=10, int_downsize=1):
    """Integrate a stationary velocity field via scaling and squaring.

    Args:
        svf: Tensor of shape (B, 3, H, W, D).
        int_steps: Number of integration steps.
        int_downsize: Downsampling factor before integration.

    Returns:
        Deformation field of shape (B, 3, H, W, D).
    """
    assert svf.dim() == 5 and svf.shape[1] == 3
    device = svf.device

    if int_downsize > 1:
        resize = ResizeTransform(int_downsize, ndims=3).to(device)
        fullsize = ResizeTransform(1 / int_downsize, ndims=3).to(device)
        svf_down = resize(svf)
    else:
        svf_down = svf
        def fullsize(x):
            return x

    integrator = VecInt(inshape=svf_down.shape[2:], nsteps=int_steps).to(device)
    deformation_field = integrator(svf_down)

    deformation_field = fullsize(deformation_field)
    return deformation_field


def jacobian_determinant_torch(flow):
    """Compute Jacobian determinant of a 3D SVF.

    Args:
        flow: Tensor (B, 3, H, W, D) — stationary velocity field.

    Returns:
        Tensor (B, H-1, W-1, D-1) — Jacobian determinant at each voxel.
    """
    flow = integrate_svf(flow)
    B, C, H, W, D = flow.shape

    grid = create_voxel_identity_grid((H, W, D), flow.device)
    mapping = flow + grid

    dfx = mapping[:, :, :, 1:, :] - mapping[:, :, :, :-1, :]
    dfy = mapping[:, :, 1:, :, :] - mapping[:, :, :-1, :, :]
    dfz = mapping[:, :, :, :, 1:] - mapping[:, :, :, :, :-1]

    dfx = dfx[:, :, :-1, :, :-1]
    dfy = dfy[:, :, :, :-1, :-1]
    dfz = dfz[:, :, :-1, :-1, :]

    J = torch.stack([dfy, dfx, dfz], dim=-1)
    J = J.permute(0, 2, 3, 4, 1, 5)  # (B, H-1, W-1, D-1, 3, 3)

    jac_det = (
        J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1])
        - J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0])
        + J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0])
    )
    return jac_det


def channelwise_epe_loss(predicted_field, target_field):
    """Per-channel L1 end-point error for 3D deformation fields.

    Returns:
        tuple: (total_epe, x_epe, y_epe, z_epe)
    """
    diff = predicted_field - target_field
    x_epe = torch.mean(torch.abs(diff[:, 0, ...]))
    y_epe = torch.mean(torch.abs(diff[:, 1, ...]))
    z_epe = torch.mean(torch.abs(diff[:, 2, ...]))
    total_epe = (x_epe + y_epe + z_epe) / 3.0
    return total_epe, x_epe, y_epe, z_epe


def angular_error_loss(predicted_field, target_field, eps=1e-6):
    """Angular error between predicted and target 3D deformation fields.

    Returns:
        Mean angular error in radians.
    """
    batch_size = predicted_field.shape[0]
    pred_flat = predicted_field.view(batch_size, 3, -1)
    target_flat = target_field.view(batch_size, 3, -1)

    pred_mag = torch.sqrt(torch.sum(pred_flat**2, dim=1) + eps)
    target_mag = torch.sqrt(torch.sum(target_flat**2, dim=1) + eps)
    dot_product = torch.sum(pred_flat * target_flat, dim=1)
    cos_sim = torch.clamp(dot_product / (pred_mag * target_mag), -1.0 + eps, 1.0 - eps)

    return torch.mean(torch.acos(cos_sim))
