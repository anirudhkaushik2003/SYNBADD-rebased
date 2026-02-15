"""DAFT (Dynamic Affine Feature map Transform) blocks.

Implements FiLM-style conditioning for 3D convolutional networks,
adapted from the ZeCatNet architecture.
"""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    padding = 1 if kernel_size != 1 else 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class FilmBase(nn.Module, metaclass=ABCMeta):
    """Abstract base class for FiLM-style conditioning (Perez et al.)."""

    def __init__(self, in_channels, out_channels, bn_momentum, stride,
                 ndim_non_img, location, activation, scale, shift):
        super().__init__()

        if location not in set(range(5)):
            raise ValueError(f"Invalid location: {location}")
        if activation not in {"tanh", "sigmoid", "linear"}:
            raise ValueError(f"Invalid activation: {activation}")
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(f"scale and shift must be bool, at least one True: scale={scale}, shift={shift}")

        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, affine=(location != 3))
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

        self.location = location
        if self.location == 2 and self.downsample is None:
            raise ValueError("location=2 with no downsampling is equivalent to location=1")

        self.film_dims = in_channels if location in {0, 1, 2} else out_channels

        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        else:
            self.scale_activation = None

    @abstractmethod
    def rescale_features(self, feature_map, x_aux):
        """Recalibrate feature map using auxiliary input."""

    def forward(self, feature_map, x_aux):
        if self.location == 0:
            feature_map = self.rescale_features(feature_map, x_aux)
        residual = feature_map

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)
        if self.location == 2:
            feature_map = self.rescale_features(feature_map, x_aux)

        out = self.conv1(feature_map)
        out = self.bn1(out)

        if self.location == 3:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)


class _FilmRescaleMixin:
    """Shared rescale logic for FilmBlock and DAFTBlock."""

    def _apply_rescale(self, feature_map, attention):
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention.view(*attention.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention.view(*attention.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(f"Invalid scale/shift: {self.scale}, {self.shift}")
        return (v_scale * feature_map) + v_shift


class DAFTBlock(FilmBase, _FilmRescaleMixin):
    """Dynamic Affine Feature map Transform block.

    Uses global average pooling of the feature map concatenated with
    auxiliary input to compute scale and shift parameters.
    """

    def __init__(self, in_channels, out_channels, bn_momentum=0.1, stride=2,
                 ndim_non_img=15, location=0, activation="linear",
                 scale=True, shift=True, bottleneck_dim=7):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            bn_momentum=bn_momentum, stride=stride, ndim_non_img=ndim_non_img,
            location=location, activation=activation, scale=scale, shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims

        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)
        attention = self.aux(squeeze)
        return self._apply_rescale(feature_map, attention)
