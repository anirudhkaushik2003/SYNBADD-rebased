# =============================================================================
# File: DiffeomorphicRegistration.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffeomorphicRegistration(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, 3, padding=1)
        )

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        displacement = self.encoder(x)
        return displacement

    def create_deformation_grid(self, displacement, shape):
        B, C, H, W, D = shape
        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0).repeat(B, 1, 1).to(displacement.device),
                             size=shape, align_corners=True)
        grid = grid + displacement.permute(0, 2, 3, 4, 1)
        return grid

    def warp(self, moving, displacement):
        grid = self.create_deformation_grid(displacement, moving.shape)
        warped = F.grid_sample(moving, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

# 🔄 Loss Functions
def similarity_loss(warped, target):
    return F.mse_loss(warped, target)

def smoothness_loss(displacement):
    dx = torch.abs(displacement[:, :, 1:, :, :] - displacement[:, :, :-1, :, :])
    dy = torch.abs(displacement[:, :, :, 1:, :] - displacement[:, :, :, :-1, :])
    dz = torch.abs(displacement[:, :, :, :, 1:] - displacement[:, :, :, :, :-1])
    return (dx.mean() + dy.mean() + dz.mean()) / 3