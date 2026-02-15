# =============================================================================
# File: models.py
# Purpose: Auxiliary model definitions (including VAE components and latent regressors) shared by training/evaluation scripts.
# Prerequisites: Imported by utils.py and training/testing entrypoints.
# Usually Run Next: Used as backbone modules during train/test runs.
# Model Components Used: Contains LatentRegressor and VAE3D_UNetSkip; latest flagship path typically uses VAE from get_vae_model plus transformer temporal encoder.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 1024),
            nn.Tanh(),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, t, z):
        # z: [B, latent_dim], t: scalar
        t_ = torch.ones(z.size(0), 1, device=z.device) * t  # [B, 1]
        zt = torch.cat([z, t_], dim=-1)
        return self.net(zt)

class LatentRegressor(nn.Module):
    def __init__(self, seq_len=5):
        super().__init__()
        self.seq_len = seq_len

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.seq_len * 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, z_known):  # [B, 4, 512]
        return self.net(z_known)


class VAE3D_UNetSkip(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super(VAE3D_UNetSkip, self).__init__()
        self.latent_dim = latent_dim

        # ===== Encoder (Downsampling) =====
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )      # [32, 128, 128, 128]
        self.enc2 = self.conv_block(32, 64)               # [64, 64, 64, 64]
        self.enc3 = self.conv_block(64, 128)              # [128, 32, 32, 32]
        self.enc4 = self.conv_block(128, 256)             # [256, 16, 16, 16]
        self.enc5 = self.conv_block(256, 512)             # [512, 8, 8, 8]
        self.enc6 = self.conv_block(512, 512)             # [512, 4, 4, 4]

        # Flatten and Latent
        self.fc_mu = nn.Linear(512 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        # ===== Decoder (Upsampling with skip connections) =====
        self.up1 = self.up_block(512, 512)                # [512, 8, 8, 8]
        self.up2 = self.up_block(1024, 256)               # [256, 16, 16, 16]
        self.up3 = self.up_block(512, 128)                # [128, 32, 32, 32]
        self.up4 = self.up_block(256, 64)                 # [64, 64, 64, 64]
        self.up5 = self.up_block(128, 32)                 # [32, 128, 128, 128]

        self.final = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()  # Or Sigmoid depending on your data

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        e1 = self.enc1(x)  # [B, 32, 128, 128, 128]
        e2 = self.enc2(e1) # [B, 64, 64, 64, 64]
        e3 = self.enc3(e2) # [B, 128, 32, 32, 32]
        e4 = self.enc4(e3) # [B, 256, 16, 16, 16]
        e5 = self.enc5(e4) # [B, 512, 8, 8, 8]
        e6 = self.enc6(e5) # [B, 512, 4, 4, 4]

        flat = e6.view(e6.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, [e1, e2, e3, e4, e5]

    def decode(self, z, skips):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 4, 4, 4)

        x = self.up1(x)                     # [B, 512, 8, 8, 8]
        x = self.up2(torch.cat([x, skips[4]], dim=1))  # [B, 256, 16, 16, 16]
        x = self.up3(torch.cat([x, skips[3]], dim=1))  # [B, 128, 32, 32, 32]
        x = self.up4(torch.cat([x, skips[2]], dim=1))  # [B, 64, 64, 64, 64]
        x = self.up5(torch.cat([x, skips[1]], dim=1))  # [B, 32, 128, 128, 128]
        out = self.final(torch.cat([x, skips[0]], dim=1))  # [B, 3, 128, 128, 128]
        return self.activation(out)

    def forward(self, x, mode="default", skips_d=None):
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x, skips_d)
        elif mode == "default":
            mu, logvar, skips = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, skips)
            return recon, mu, logvar

class TimeConditionedVAE3D_UNetSkip(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, num_timepoints=5, time_embedding_dim=64):
        super(TimeConditionedVAE3D_UNetSkip, self).__init__()
        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim
        self.total_latent_dim = latent_dim  # The VAE latent z is still latent_dim

        # ===== Encoder =====
        self.enc1 = nn.Sequential(nn.Conv3d(in_channels, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU())
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)
        self.enc6 = self.conv_block(512, 512)

        self.fc_mu = nn.Linear(512 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4 * 4, latent_dim)

        # ===== Time Embedding =====
        self.time_embedding = nn.Embedding(num_timepoints, time_embedding_dim)

        # ===== Cross-Attention Layer =====
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=1, batch_first=True)
        self.attn_ln = nn.LayerNorm(latent_dim)  # Optional normalization

        # ===== Time Classifier (for supervision) =====
        self.time_cls_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_timepoints)
        )

        # ===== Decoder =====
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        self.up1 = self.up_block(512, 512)
        self.up2 = self.up_block(1024, 256)
        self.up3 = self.up_block(512, 128)
        self.up4 = self.up_block(256, 64)
        self.up5 = self.up_block(128, 32)

        self.final = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        flat = e6.view(e6.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, [e1, e2, e3, e4, e5]

    def fuse_time(self, z, time_idx):
        # z: (B, D), time_idx: (B,)
        B, D = z.shape
        time_embed = self.time_embedding(time_idx)  # (B, D_time)

        # Expand z and time for attention: Q = z, K = V = time_embed
        q = z.unsqueeze(1)  # (B, 1, D)
        k = v = time_embed.unsqueeze(1)  # (B, 1, D_time), project to match D if needed

        # Project time embedding to match latent dim if needed
        if self.time_embedding_dim != self.latent_dim:
            if not hasattr(self, 'time_proj'):
                self.time_proj = nn.Linear(self.time_embedding_dim, self.latent_dim).to(z.device)
            k = v = self.time_proj(time_embed).unsqueeze(1)  # (B, 1, D)

        fused, _ = self.cross_attn(q, k, v)  # (B, 1, D)
        fused = self.attn_ln(fused.squeeze(1) + z)  # Residual + LN
        return fused  # (B, D)

    def decode(self, z, skips):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 4, 4, 4)
        x = self.up1(x)
        x = self.up2(torch.cat([x, skips[4]], dim=1))
        x = self.up3(torch.cat([x, skips[3]], dim=1))
        x = self.up4(torch.cat([x, skips[2]], dim=1))
        x = self.up5(torch.cat([x, skips[1]], dim=1))
        out = self.final(torch.cat([x, skips[0]], dim=1))
        return self.activation(out)

    def forward(self, x, time_idx):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_fused = self.fuse_time(z, time_idx)
        recon = self.decode(z_fused, skips)
        time_logits = self.time_cls_head(z_fused)
        return recon, mu, logvar, time_logits
    

class FactorizedTimeConditionedVAE3D_UNetSkip(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, num_timepoints=5, time_embedding_dim=64):
        super(TimeConditionedVAE3D_UNetSkip, self).__init__()
        self.latent_dim = latent_dim
        self.latent_dim_time = latent_dim // 2
        self.latent_dim_content = latent_dim - self.latent_dim_time
        self.time_embedding_dim = time_embedding_dim
        self.num_timepoints = num_timepoints

        # ===== Encoder =====
        self.enc1 = nn.Sequential(nn.Conv3d(in_channels, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU())
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)
        self.enc6 = self.conv_block(512, 512)

        self.flatten_dim = 512 * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ===== Time Embedding =====
        self.time_embedding = nn.Embedding(num_timepoints, time_embedding_dim)
        self.time_proj = nn.Linear(time_embedding_dim, self.latent_dim_time)

        # ===== Time Classifier =====
        self.time_cls_head = nn.Sequential(
            nn.Linear(self.latent_dim_time, 64),
            nn.ReLU(),
            nn.Linear(64, num_timepoints)
        )

        # ===== Decoder =====
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.up1 = self.up_block(512, 512)
        self.up2 = self.up_block(1024, 256)
        self.up3 = self.up_block(512, 128)
        self.up4 = self.up_block(256, 64)
        self.up5 = self.up_block(128, 32)

        self.final = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        flat = e6.view(e6.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, [e1, e2, e3, e4, e5]

    def fuse_time(self, z_time, time_idx):
        time_embed = self.time_embedding(time_idx)  # (B, D_time_embed)
        time_embed_proj = self.time_proj(time_embed)  # (B, latent_dim_time)
        return z_time + time_embed_proj  # simple residual conditioning

    def decode(self, z, skips):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 4, 4, 4)
        x = self.up1(x)
        x = self.up2(torch.cat([x, skips[4]], dim=1))
        x = self.up3(torch.cat([x, skips[3]], dim=1))
        x = self.up4(torch.cat([x, skips[2]], dim=1))
        x = self.up5(torch.cat([x, skips[1]], dim=1))
        out = self.final(torch.cat([x, skips[0]], dim=1))
        return self.activation(out)

    def forward(self, x, time_idx):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_time = z[:, :self.latent_dim_time]
        z_content = z[:, self.latent_dim_time:]

        z_time_fused = self.fuse_time(z_time, time_idx)
        z_fused = torch.cat([z_time_fused, z_content], dim=1)

        recon = self.decode(z_fused, skips)
        time_logits = self.time_cls_head(z_time_fused)

        return recon, mu, logvar, time_logits, z_time, z_content
