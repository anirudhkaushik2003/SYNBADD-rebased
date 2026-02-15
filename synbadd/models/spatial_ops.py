"""Spatial operation modules for deformation-based image generation.

Provides differentiable spatial transformers, vector field integration,
and safe upsampling utilities adapted from VoxelMorph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeUpsample(nn.Module):
    """Upsampling module that handles large 3D volumes by processing in chunks
    when the output would exceed int32 element limits."""

    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        batch_size, channels, d, h, w = x.shape

        output_size = (
            d * self.scale_factor,
            h * self.scale_factor,
            w * self.scale_factor,
        )
        total_elements = batch_size * channels * output_size[0] * output_size[1] * output_size[2]

        if total_elements < 2**31:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        # Process in chunks along the depth dimension
        chunk_size = max(1, d // 4)
        output_chunks = []
        for i in range(0, d, chunk_size):
            end_idx = min(i + chunk_size, d)
            chunk = x[:, :, i:end_idx, :, :]
            upsampled_chunk = F.interpolate(
                chunk, scale_factor=self.scale_factor, mode=self.mode,
            )
            output_chunks.append(upsampled_chunk)

        return torch.cat(output_chunks, dim=2)


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer that warps a source image by a displacement field."""

    def __init__(self, size, mode='bilinear', get_unnormed=False):
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        self.register_buffer('grid', grid)
        self.get_unnormed = get_unnormed

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        unnormed_locs = torch.clone(new_locs)

        # Normalize grid values to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if self.get_unnormed:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), unnormed_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrates a vector field via scaling and squaring."""

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
    """Resize a transform, which involves resizing the vector field *and*
    rescaling it so that the displacement magnitudes remain consistent."""

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
            # Resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # Multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        # No-op when factor == 1
        return x
