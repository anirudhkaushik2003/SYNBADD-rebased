"""MONAI transform presets for training and evaluation."""

import monai.transforms as t
from monai.transforms import NormalizeIntensity


def train_transforms(aug_p: float = 0.8):
    """Augmented transforms for training."""
    return t.Compose([
        t.ToTensor(),
        t.Rand3DElastic(
            sigma_range=(0.01, 1),
            magnitude_range=(0, 1),
            prob=aug_p,
            rotate_range=(0.18, 0.18, 0.18),
            translate_range=(4, 4, 4),
            scale_range=(0.10, 0.10, 0.10),
            spatial_size=None,
            padding_mode="border",
        ),
        t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
        t.RandAdjustContrast(prob=aug_p),
        t.RandGaussianNoise(prob=aug_p),
        t.NormalizeIntensity(),
    ])


def eval_transforms():
    """Transforms for validation/testing (no augmentation)."""
    return t.Compose([
        t.ToTensor(),
        NormalizeIntensity(),
    ])
