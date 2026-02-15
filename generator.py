# =============================================================================
# File: generator.py
# Purpose: Core generator architectures used across SYNBADD experiments (context, SVF, regression/diffeomorphic variants).
# Prerequisites: Imported by training/testing entrypoints; relies on torch and losses utilities.
# Usually Run Next: Used by train_VAE_SYNBADD_svf.py, test_model_subjectwise.py, robustness_exp.py, and related train/test scripts.
# Model Components Used: Defines active generator classes including SVFRegressorDiffeoGenerator used by latest subjectwise pipeline.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from daft_blocks import DAFTBlock
from losses import jacobian_determinant_torch

'''
This file contains the code pertinent to creating a GAN generator that produces diffeomorphisms.
This adapts the architecture from Adrian Dalca et.al, see https://github.com/voxelmorph/voxelmorph
if you use it, cite their paper as well "An Unsupervised Learning Model for Deformable Medical Image Registration"
'''


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

class PhysicalSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer that models physical tissue properties during deformation
    """

    def __init__(self, size, mode='bilinear', tissue_model='density'):
        super().__init__()

        self.mode = mode
        self.tissue_model = tissue_model  # 'density', 'elasticity', or 'relaxation'

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # Apply standard transformation
        warped = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        
        # Calculate Jacobian determinant using the imported function
        # Note: returns a tensor with size (B, H-1, W-1, D-1)
        jacobian_det = jacobian_determinant_torch(flow)
        
        # Skip physical modeling if we're not using it
        if self.tissue_model == 'none':
            return warped
        
        # Resize jacobian_det to match warped image spatial dimensions
        # First, add a channel dimension for interpolation
        jacobian_det = jacobian_det.unsqueeze(1)  # (B, 1, H-1, W-1, D-1)
        
        # Interpolate to match warped image size
        jacobian_det = F.interpolate(
            jacobian_det, 
            size=warped.shape[2:],
            mode='trilinear',
            align_corners=True
        )  # (B, 1, H, W, D)
        
        # Remove channel dimension for calculations
        jacobian_det = jacobian_det.squeeze(1)  # (B, H, W, D)
        
        # Apply physical tissue model based on deformation
        if self.tissue_model == 'density':
            # Conservation of tissue density: intensity changes inversely with volume
            intensity_mod = 1.0 / torch.clamp(jacobian_det, min=0.1)
            intensity_mod = intensity_mod.unsqueeze(1)  # Add channel dimension
            warped = warped * intensity_mod
            
        elif self.tissue_model == 'elasticity':
            # Non-linear elasticity model: tissue properties change with compression
            elasticity_factor = torch.pow(torch.clamp(jacobian_det, 0.1, 1.0), 0.5)
            elasticity_factor = elasticity_factor.unsqueeze(1)  # Add channel dimension
            warped = warped * elasticity_factor
            
        elif self.tissue_model == 'relaxation':
            # Tissue T1/T2 relaxation properties change with compression
            relaxation_factor = torch.sigmoid(5 * (jacobian_det - 0.8))
            relaxation_factor = relaxation_factor.unsqueeze(1)  # Add channel dimension
            warped = warped * relaxation_factor
        
        return warped

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


class Unet(nn.Module):
    def __init__(self, inshape, c_dim, nb_features=None, nb_levels=None, feat_mult=1, in_channels=1, out_channels=1):
        super().__init__()

        self.c_dim = c_dim  # Conditioning dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if nb_features is None:
            nb_features = default_unet_features()

        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = SafeUpsample(scale_factor=2, mode='nearest')

        # Encoder path
        prev_nf = in_channels
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        self.nf_encoder = prev_nf

        # Decoder path (without DAFT layers)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        self.decoder_nf = prev_nf

        # Extras block (DAFT layers moved here)
        prev_nf += in_channels
        self.extras = nn.ModuleList()
        self.daft_layers = nn.ModuleList()  # DAFT layers now created along with extras

        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))

            # Create DAFTBlock at the same stage as extras
            daft_layer = DAFTBlock(
                in_channels=prev_nf,  # Match the number of input channels dynamically
                out_channels=prev_nf,
                bn_momentum=0.1,
                stride=1,
                ndim_non_img=c_dim,
                location=3,  # Apply DAFT after BatchNorm in extras
                activation="linear",
                scale=True,
                shift=True,
                bottleneck_dim=7
            )
            self.daft_layers.append(daft_layer)

            prev_nf = nf  # Update for next layer

    def forward(self, x, x_aux):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        x = x_enc.pop()
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        for i, layer in enumerate(self.extras):

            x = self.daft_layers[i](x, x_aux)
            x = layer(x)

        return x
    

class Unet_NoDAFT(nn.Module):
    """
    A unet architecture that only takes original images as input without condition concatenation.
    Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, in_channels=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            in_channels: Number of input channels. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample = SafeUpsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # Now using only in_channels without c_dim
        prev_nf = in_channels
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        self.nf_encoder = prev_nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        self.decoder_nf = prev_nf

        # configure extra decoder convolutions (no up-sampling)
        # Modified to only add in_channels without c_dim
        prev_nf += in_channels
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        """
        Parameters:
            x: Input tensor (original image)
        
        Returns:
            Output feature map
        """
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
class Warper(nn.Module):
    def __init__(self, latent_input_dim, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=False, in_channels=3, using_svf=True, using_daft=True, using_unet=True):

        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim
        self.using_svf = using_svf

        if not using_daft:
            print("\n\nGenerator is NOT using DAFT condition, only a simple UNET is being used ...", end='\n\n')

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # convert from latent_input_dim to in_channels, inshape to match the input of the unet
        if self.using_svf:
            H, W, D = inshape
            self.project = nn.Sequential(
                nn.Linear(latent_input_dim, in_channels * H//8 * W//8 * D//8),
                # reshape to c, H//8, W//8, D//8
                nn.Unflatten(1, (in_channels, H//8, W//8, D//8)),
                # upsample to H, W, D
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H//4, W//4, D//4
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H//2, W//2, D//2
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H, W, D
                nn.ReLU(inplace=True),
            )

        # configure core unet model
        add_factor = 1 if self.using_svf else 0
        add_factor = 0

        self.using_daft = using_daft
        self.using_unet = using_unet

        if using_unet:
            if using_daft:
                self.unet_model = Unet(
                    inshape,
                    c_dim,
                    nb_features=nb_unet_features,
                    nb_levels=nb_unet_levels,
                    feat_mult=unet_feat_mult,
                    in_channels=in_channels+add_factor,
                )
            else:
                self.unet_model = Unet_NoDAFT(
                    inshape,
                    nb_features=nb_unet_features,
                    nb_levels=nb_unet_levels,
                    feat_mult=unet_feat_mult,
                    in_channels=in_channels+add_factor,
                )
        else:
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.projection_layer = Conv(
                in_channels + add_factor, 
                in_channels + add_factor,  # Same number of output channels to maintain shape
                kernel_size=3, 
                padding=1
            )

        if using_unet:
            # configure unet to flow field layer
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

            # init flow layer with small weights and bias
            self.flow.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow.weight.shape))
            self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

            if use_probs:
                self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
                # init flow layer with small weights and bias
                self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
                self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, contextualized_scan, return_fields: bool = False):
        """
        Parameters:
            source:            Tensor (B, C, H, W, D)
            contextualized_scan: Tensor (B, 3, H?, W?, D?)  -> averaged SVF
            return_fields:     If True, also returns {'svf': ..., 'disp': ...}

        Returns:
            warped: Tensor (B, C, H, W, D)
            disp:   Tensor (B, 3, H, W, D)  # integrated displacement (φ(x) - x)
            (optionally) dict with 'svf' and 'disp'
        """
        
        svf = contextualized_scan

        preint_svf = svf  # keep for inspection / loss terms

        # 1) Optional downsample for faster integration
        if self.resize is not None:
            svf_ds = self.resize(svf)     # IMPORTANT: ResizeTransform must rescale vector magnitudes!
        else:
            svf_ds = svf

        # 2) Integrate SVF -> displacement (scaling & squaring)
        if self.integrate is not None:
            disp_ds = self.integrate(svf_ds)
        else:
            # If integrate is None, we assume input is already a displacement (not your case, but safe fallback)
            disp_ds = svf_ds

        # 3) Upsample displacement back to full resolution
        if self.fullsize is not None:
            disp = self.fullsize(disp_ds)  # same note: magnitudes must be correctly scaled in ResizeTransform
        else:
            disp = disp_ds

        # 4) Warp source with displacement
        warped = self.transformer(source, disp)

       
        return warped, contextualized_scan
        
class SVFRegressorDiffeoGenerator(nn.Module):
    def __init__(self, latent_input_dim, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=False, in_channels=3, using_svf=True, using_daft=True, using_unet=True):
        """
        Parameters:
            latent_input_dim: Latent input dimension from transformer model e.g. 512
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim
        self.using_svf = using_svf

        if not using_daft:
            print("\n\nGenerator is NOT using DAFT condition, only a simple UNET is being used ...", end='\n\n')

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # convert from latent_input_dim to in_channels, inshape to match the input of the unet
        if self.using_svf:
            H, W, D = inshape
            self.project = nn.Sequential(
                nn.Linear(latent_input_dim, in_channels * H//8 * W//8 * D//8),
                # reshape to c, H//8, W//8, D//8
                nn.Unflatten(1, (in_channels, H//8, W//8, D//8)),
                # upsample to H, W, D
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H//4, W//4, D//4
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H//2, W//2, D//2
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H, W, D
                nn.ReLU(inplace=True),
            )

        # configure core unet model
        add_factor = 1 if self.using_svf else 0
        add_factor = 0

        self.using_daft = using_daft
        self.using_unet = using_unet

        if using_unet:
            if using_daft:
                self.unet_model = Unet(
                    inshape,
                    c_dim,
                    nb_features=nb_unet_features,
                    nb_levels=nb_unet_levels,
                    feat_mult=unet_feat_mult,
                    in_channels=in_channels+add_factor,
                )
            else:
                self.unet_model = Unet_NoDAFT(
                    inshape,
                    nb_features=nb_unet_features,
                    nb_levels=nb_unet_levels,
                    feat_mult=unet_feat_mult,
                    in_channels=in_channels+add_factor,
                )
        else:
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.projection_layer = Conv(
                in_channels + add_factor, 
                in_channels + add_factor,  # Same number of output channels to maintain shape
                kernel_size=3, 
                padding=1
            )

        if using_unet:
            # configure unet to flow field layer
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

            # init flow layer with small weights and bias
            self.flow.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow.weight.shape))
            self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

            if use_probs:
                self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
                # init flow layer with small weights and bias
                self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
                self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, contextualized_scan, condition, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            contextualized_scan: Contextualized scan from Transformer model.
            registration: Return transformed image and flow. Default is False.
        '''
        
        # # Use contextualized_scan for generating deformation field
        # if self.using_svf:
        #     if contextualized_scan.shape[2:] != source.shape[2:]:
        #         contextualized_scan = F.interpolate(contextualized_scan, size=source.shape[2:], mode='trilinear', align_corners=True)

        #     # Concatenate contextualized scan and source along the channel dimension
        #     # contextualized_scan = torch.cat((contextualized_scan, source), dim=1)

        # if self.using_unet:
        #     if self.using_daft:
        #         shape = self.unet_model(contextualized_scan, condition)
        #     else:
        #         shape = self.unet_model(contextualized_scan)
                
        #     # Transform into flow field
        #     flow_field = self.flow(shape)

        #     # Resize flow for integration
        #     pos_flow = flow_field
        #     if self.resize:
        #         pos_flow = self.resize(pos_flow)

        #     preint_flow = pos_flow

        #     # Integrate to produce diffeomorphic warp
        #     if self.integrate:
        #         pos_flow = self.integrate(pos_flow)

        #         # Resize to final resolution
        #         if self.fullsize:
        #             pos_flow = self.fullsize(pos_flow)

        #     # jac_det = jacobian_determinant_torch(pos_flow)
        #     # print("Jacobian mean:", jac_det.mean().item(), "min:", jac_det.min().item(), "neg%:", (jac_det < 0).float().mean().item() * 100)

        # else:
        pos_flow = self.projection_layer(contextualized_scan)
        preint_flow = contextualized_scan
        # Warp source image with flow field
        y_source = self.transformer(source, pos_flow)

        # Return warped image and flow field
        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow
    
class SVFTransformerDiffeoGenerator(nn.Module):
    def __init__(self, latent_input_dim, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=False, in_channels=3, using_svf=True):
        """
        Parameters:
            latent_input_dim: Latent input dimension from transformer model e.g. 512
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim
        self.using_svf = using_svf

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # convert from latent_input_dim to in_channels, inshape to match the input of the unet
        if self.using_svf:
            H, W, D = inshape
            self.project = nn.Sequential(
                nn.Linear(latent_input_dim, in_channels * H//8 * W//8 * D//8),
                # reshape to c, H//8, W//8, D//8
                nn.Unflatten(1, (in_channels, H//8, W//8, D//8)),
                # upsample to H, W, D
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H//4, W//4, D//4
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H//2, W//2, D//2
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # H, W, D
                nn.ReLU(inplace=True),
            )

        # configure core unet model
        add_factor = 1 if self.using_svf else 0
        self.unet_model = Unet(
            inshape,
            c_dim,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            in_channels=in_channels+add_factor,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_probs:
            self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
            # init flow layer with small weights and bias
            self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
            self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        # self.transformer = PhysicalSpatialTransformer(inshape, mode='bilinear', tissue_model='elasticity')

    def forward(self, source, contextualized_scan, condition, registration=True):
        '''
        Parameters:
            source: Source image tensor.
            contextualized_scan: Contextualized scan from ViViT model.
            registration: Return transformed image and flow. Default is False.
        '''
        
        # Use contextualized_scan for generating deformation field
        if self.using_svf:
            contextualized_scan = self.project(contextualized_scan)
            if contextualized_scan.shape[2:] != source.shape[2:]:
                contextualized_scan = F.interpolate(contextualized_scan, size=source.shape[2:], mode='trilinear', align_corners=True)

            # Concatenate contextualized scan and source along the channel dimension
            contextualized_scan = torch.cat((contextualized_scan, source), dim=1)

        shape = self.unet_model(contextualized_scan, condition)
        # Transform into flow field
        flow_field = self.flow(shape)

        # Resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # Integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # Resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # jac_det = jacobian_determinant_torch(pos_flow)
        # print("Jacobian mean:", jac_det.mean().item(), "min:", jac_det.min().item(), "neg%:", (jac_det < 0).float().mean().item() * 100)

        # Warp source image with flow field
        y_source = self.transformer(source, pos_flow)

        # Return warped image and flow field
        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow

class SVFContextDiffeoGenerator(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=True, in_channels=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            c_dim,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            in_channels=in_channels
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_probs:
            self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
            # init flow layer with small weights and bias
            self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
            self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, contextualized_scan, condition, registration=True):
        '''
        Parameters:
            source: Source image tensor.
            contextualized_scan: Contextualized scan from ViViT model.
            registration: Return transformed image and flow. Default is False.
        '''
        
        # Use contextualized_scan for generating deformation field
        if contextualized_scan.shape[2:] != source.shape[2:]:
            contextualized_scan = F.interpolate(contextualized_scan, size=source.shape[2:], mode='trilinear', align_corners=True)
        shape = self.unet_model(contextualized_scan, condition)

        # Transform into flow field
        flow_field = self.flow(shape)

        # Resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # Integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # Resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)


        # Warp source image with flow field
        y_source = self.transformer(source, pos_flow)

        # Return warped image and flow field
        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow
        
class ContextDiffeoGenerator(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=True, in_channels=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            c_dim,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            in_channels=in_channels
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-4).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_probs:
            self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
            # init flow layer with small weights and bias
            self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
            self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, contextualized_scan, condition, registration=True):
        '''
        Parameters:
            source: Source image tensor.
            contextualized_scan: Contextualized scan from ViViT model.
            registration: Return transformed image and flow. Default is False.
        '''
        
        # Use contextualized_scan for generating deformation field
        shape = self.unet_model(contextualized_scan, condition)

        # Transform into flow field
        flow_field = self.flow(shape)

        # Resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # Integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # Resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # Warp source image with flow field
        y_source = self.transformer(source, pos_flow)

        # Return warped image and flow field
        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow

class CondDiffeoGenerator(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=True, in_channels=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            c_dim,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            in_channels=in_channels
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_probs:
            self.flow_logsigma = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
            # init flow layer with small weights and bias
            self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
            self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target label to which to convert the image to.
            registration: Return transformed image and flow. Default is False.
        '''

        target = target.view(target.size(0), self.c_dim, 1, 1, 1)
        target = target.repeat(1, 1, source.size(2), source.size(3), source.size(4))
        
        to_unet = torch.cat((source, target), 1)

        # concatenate inputs and propagate unet
        shape = self.unet_model(to_unet)

        # transform into flow field
        flow_field = self.flow(shape)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        # return non-integrated flow field if training
        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

