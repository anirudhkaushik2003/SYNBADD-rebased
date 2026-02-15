"""VAE3D with U-Net skip connections for 3D medical image encoding/decoding."""

import torch
import torch.nn as nn

from synbadd.models.registry import register_model


@register_model("VAE3D_UNetSkip")
class VAE3D_UNetSkip(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )  # [32, 128, 128, 128]
        self.enc2 = self._conv_block(32, 64)    # [64, 64, 64, 64]
        self.enc3 = self._conv_block(64, 128)   # [128, 32, 32, 32]
        self.enc4 = self._conv_block(128, 256)  # [256, 16, 16, 16]
        self.enc5 = self._conv_block(256, 512)  # [512, 8, 8, 8]
        self.enc6 = self._conv_block(512, 512)  # [512, 4, 4, 4]

        # Flatten and Latent
        self.fc_mu = nn.Linear(512 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        # Decoder (Upsampling with skip connections)
        self.up1 = self._up_block(512, 512)   # [512, 8, 8, 8]
        self.up2 = self._up_block(1024, 256)  # [256, 16, 16, 16]
        self.up3 = self._up_block(512, 128)   # [128, 32, 32, 32]
        self.up4 = self._up_block(256, 64)    # [64, 64, 64, 64]
        self.up5 = self._up_block(128, 32)    # [32, 128, 128, 128]

        self.final = nn.Conv3d(64, in_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
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
