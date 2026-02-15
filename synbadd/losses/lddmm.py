"""LDDMM-style loss using pretrained VoxelMorph model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from synbadd.losses.deformation import jacobian_determinant_torch


class VxmLDDMMLoss(nn.Module):
    """Differentiable LDDMM loss using a pretrained VxmDense model.

    Computes a combination of multi-scale intensity similarity and
    Jacobian regularization from the VoxelMorph-estimated flow field.
    """

    def __init__(self, vxm_model, weight_jacobian=0.01, weight_intensity=1.0):
        super().__init__()
        self.vxm = vxm_model
        self.weight_jacobian = weight_jacobian
        self.weight_intensity = weight_intensity

    def multiscale_mse(self, x, y):
        """MSE averaged across 3 scales."""
        loss = F.mse_loss(x, y)
        loss += F.mse_loss(F.avg_pool3d(x, 2), F.avg_pool3d(y, 2))
        loss += F.mse_loss(F.avg_pool3d(x, 4), F.avg_pool3d(y, 4))
        return loss / 3.0

    def forward(self, target_img, generated_img):
        """Compute LDDMM loss.

        Args:
            target_img: [B, 1, H, W, D] — fixed image.
            generated_img: [B, 1, H, W, D] — moving image (generated).

        Returns:
            Scalar loss tensor.
        """
        warped_img, flow_field = self.vxm(generated_img, target_img)

        intensity_loss = self.multiscale_mse(warped_img, target_img)

        jacobian = jacobian_determinant_torch(flow_field)
        jacobian_loss = torch.mean(torch.abs(jacobian - 1.0))

        return self.weight_intensity * intensity_loss + self.weight_jacobian * jacobian_loss
