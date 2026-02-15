"""Composite loss assembled from config coefficients."""

import torch
import torch.nn as nn

from synbadd.losses.deformation import (
    angular_error_loss,
    channelwise_epe_loss,
    jacobian_determinant_torch,
)
from synbadd.losses.regularization import r1_reg, smooth_loss_l2
from synbadd.losses.similarity import ssim_loss


class CompositeLoss(nn.Module):
    """Weighted sum of all SYNBADD losses.

    Computes individual losses and returns:
        - total_loss: weighted sum for backprop
        - loss_dict: individual loss values for logging
    """

    def __init__(self, loss_cfg, lddmm_loss=None):
        super().__init__()
        self.w = loss_cfg  # LossConfig dataclass
        self.lddmm_loss = lddmm_loss

    def forward(self, generated_img, target_img, synthetic_field,
                target_svf=None, disc_out=None, x_real=None):
        """Compute all losses.

        Args:
            generated_img: [B, 1, H, W, D] — generated scan.
            target_img: [B, 1, H, W, D] — real target scan.
            synthetic_field: [B, 3, H, W, D] — generated SVF.
            target_svf: [B, 3, H, W, D] — ground-truth SVF (optional).
            disc_out: discriminator output (optional).
            x_real: real input for R1 reg (optional).

        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        total = torch.tensor(0.0, device=generated_img.device)

        # SSIM loss
        if self.w.ssim > 0:
            l_ssim = ssim_loss(generated_img, target_img)
            loss_dict["ssim"] = l_ssim.item()
            total = total + self.w.ssim * l_ssim

        # LDDMM loss
        if self.w.lddmm > 0 and self.lddmm_loss is not None:
            l_lddmm = self.lddmm_loss(target_img, generated_img)
            loss_dict["lddmm"] = l_lddmm.item()
            total = total + self.w.lddmm * l_lddmm

        # Jacobian losses (requires SVF)
        if self.w.jacobian > 0 or self.w.jac_dist > 0:
            jac_det = jacobian_determinant_torch(synthetic_field)

            if self.w.jacobian > 0:
                l_jac = torch.mean(torch.abs(jac_det[jac_det < 0]))
                if torch.isnan(l_jac):
                    l_jac = torch.tensor(0.0, device=generated_img.device)
                loss_dict["jacobian"] = l_jac.item()
                total = total + self.w.jacobian * l_jac

            if self.w.jac_dist > 0:
                l_jac_dist = torch.mean(torch.abs(jac_det - 1.0))
                loss_dict["jac_dist"] = l_jac_dist.item()
                total = total + self.w.jac_dist * l_jac_dist

        # Smoothness loss
        if self.w.smooth > 0:
            l_smooth = smooth_loss_l2(synthetic_field)
            loss_dict["smooth"] = l_smooth.item()
            total = total + self.w.smooth * l_smooth

        # EPE losses (per-channel)
        if target_svf is not None and (self.w.epe_x > 0 or self.w.epe_y > 0 or self.w.epe_z > 0):
            _, x_epe, y_epe, z_epe = channelwise_epe_loss(synthetic_field, target_svf)
            if self.w.epe_x > 0:
                loss_dict["epe_x"] = x_epe.item()
                total = total + self.w.epe_x * x_epe
            if self.w.epe_y > 0:
                loss_dict["epe_y"] = y_epe.item()
                total = total + self.w.epe_y * y_epe
            if self.w.epe_z > 0:
                loss_dict["epe_z"] = z_epe.item()
                total = total + self.w.epe_z * z_epe

        # Angular error
        if target_svf is not None and self.w.angular > 0:
            l_angular = angular_error_loss(synthetic_field, target_svf)
            loss_dict["angular"] = l_angular.item()
            total = total + self.w.angular * l_angular

        # Discriminator loss
        if self.w.disc > 0 and disc_out is not None and x_real is not None:
            l_disc = r1_reg(disc_out, x_real)
            loss_dict["disc"] = l_disc.item()
            total = total + self.w.disc * l_disc

        loss_dict["total"] = total.item()
        return total, loss_dict


def build_loss_fn(cfg, lddmm_loss=None):
    """Factory to create CompositeLoss from SynbaddConfig."""
    return CompositeLoss(cfg.training.loss, lddmm_loss=lddmm_loss)
