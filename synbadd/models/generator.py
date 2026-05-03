"""SVF-based diffeomorphic generator with UNet backbone.

Adapted from VoxelMorph (Dalca et al., "An Unsupervised Learning Model
for Deformable Medical Image Registration").
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from synbadd.models.daft_blocks import DAFTBlock
from synbadd.models.registry import register_model
from synbadd.models.spatial_ops import (
    ResizeTransform,
    SafeUpsample,
    SpatialTransformer,
    VecInt,
)

# ---------------------------------------------------------------------------
# UNet helpers
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Convolutional block with LeakyReLU for UNet."""

    def __init__(self, ndims, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))


def default_unet_features():
    return [
        [16, 32, 32, 32],         # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]


# ---------------------------------------------------------------------------
# UNet variants
# ---------------------------------------------------------------------------

class Unet(nn.Module):
    """UNet with DAFT conditioning blocks."""

    def __init__(self, inshape, c_dim, nb_features=None, nb_levels=None,
                 feat_mult=1, in_channels=1):
        super().__init__()
        self.c_dim = c_dim
        ndims = len(inshape)
        assert ndims in [1, 2, 3]

        if nb_features is None:
            nb_features = default_unet_features()
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = SafeUpsample(scale_factor=2, mode='nearest')

        # Encoder
        prev_nf = in_channels
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        self.nf_encoder = prev_nf

        # Decoder
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf
        self.decoder_nf = prev_nf

        # Extras with DAFT conditioning
        prev_nf += in_channels
        self.extras = nn.ModuleList()
        self.daft_layers = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            self.daft_layers.append(DAFTBlock(
                in_channels=prev_nf, out_channels=prev_nf,
                bn_momentum=0.1, stride=1, ndim_non_img=c_dim,
                location=3, activation="linear",
                scale=True, shift=True, bottleneck_dim=7,
            ))
            prev_nf = nf

    def forward(self, x, x_aux):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        for i, layer in enumerate(self.extras):
            x = self.daft_layers[i](x, x_aux)
            x = layer(x)
        return x


class UnetNoDAFT(nn.Module):
    """UNet without DAFT conditioning (plain architecture)."""

    def __init__(self, inshape, nb_features=None, nb_levels=None,
                 feat_mult=1, in_channels=1):
        super().__init__()
        ndims = len(inshape)
        assert ndims in [1, 2, 3]

        if nb_features is None:
            nb_features = default_unet_features()
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = SafeUpsample(scale_factor=2, mode='nearest')

        # Encoder
        prev_nf = in_channels
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        self.nf_encoder = prev_nf

        # Decoder
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf
        self.decoder_nf = prev_nf

        # Extra convs at full resolution
        prev_nf += in_channels
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        for layer in self.extras:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

@register_model("SVFRegressorDiffeoGenerator")
class SVFRegressorDiffeoGenerator(nn.Module):
    """Generator that takes a latent SVF from the temporal model, projects it
    to spatial dimensions, optionally refines via UNet, and integrates it
    into a diffeomorphic displacement field to warp the source image."""

    def __init__(self, latent_input_dim, inshape, c_dim,
                 nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1,
                 int_steps=7, int_downsize=2, bidir=False, use_probs=False,
                 in_channels=3, using_svf=True, using_daft=True, using_unet=True):
        super().__init__()

        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim
        self.using_svf = using_svf
        self.using_daft = using_daft
        self.using_unet = using_unet

        ndims = len(inshape)
        assert ndims in [1, 2, 3]

        # Latent-to-spatial projection
        if self.using_svf:
            H, W, D = inshape
            self.project = nn.Sequential(
                nn.Linear(latent_input_dim, in_channels * H // 8 * W // 8 * D // 8),
                nn.Unflatten(1, (in_channels, H // 8, W // 8, D // 8)),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            )

        # UNet or simple projection
        if using_unet:
            if using_daft:
                self.unet_model = Unet(
                    inshape, c_dim, nb_features=nb_unet_features,
                    nb_levels=nb_unet_levels, feat_mult=unet_feat_mult,
                    in_channels=in_channels,
                )
            else:
                self.unet_model = UnetNoDAFT(
                    inshape, nb_features=nb_unet_features,
                    nb_levels=nb_unet_levels, feat_mult=unet_feat_mult,
                    in_channels=in_channels,
                )
        else:
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.projection_layer = Conv(in_channels, in_channels, kernel_size=3, padding=1)

        # Flow field layer (only when using UNet)
        if using_unet:
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
            self.flow.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow.weight.shape))
            self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

            if use_probs:
                self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
                self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
                self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # Optional resize for integration
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        self.bidir = bidir

        # Integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # Spatial transformer for warping
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, contextualized_scan, condition, registration=False):
        """
        Args:
            source: Source image [B, C, H, W, D]
            contextualized_scan: SVF from temporal model [B, 3, H, W, D]
            condition: Conditioning vector for DAFT [B, c_dim]
            registration: If True, return integrated flow instead of pre-integration flow
        """
        if self.using_unet:
            # UNet path: refine contextualized scan into a flow field
            if self.using_daft:
                shape = self.unet_model(contextualized_scan, condition)
            else:
                shape = self.unet_model(contextualized_scan)

            flow_field = self.flow(shape)

            pos_flow = flow_field
            if self.resize:
                pos_flow = self.resize(pos_flow)

            preint_flow = pos_flow

            if self.integrate:
                pos_flow = self.integrate(pos_flow)
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
        else:
            # Simple projection path (no UNet)
            pos_flow = self.projection_layer(contextualized_scan)
            preint_flow = contextualized_scan

        # Warp source image
        y_source = self.transformer(source, pos_flow)

        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow
