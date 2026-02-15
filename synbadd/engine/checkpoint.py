"""Checkpoint save/load utilities preserving legacy naming conventions."""

import os
from collections import OrderedDict

import torch


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel-saved state dicts."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd


class CheckpointManager:
    """Manages saving and loading of all model components.

    Preserves the exact file naming convention used in the original repo
    so that existing checkpoints can be loaded without modification.
    """

    COMPONENT_SUFFIXES = {
        "generator": "_SYNBADD_SVF.pth",
        "vae": "_vae_SYNBADD_SVF.pth",
        "temporal": "_transformer_SYNBADD_SVF.pth",
        "regressor": "_regression_SYNBADD_SVF.pth",
        "discriminator": "_discriminator_SYNBADD_SVF.pth",
    }

    def __init__(self, checkpoint_dir, project_name, run_name):
        self.save_dir = os.path.join(checkpoint_dir, project_name)
        self.run_name = run_name
        os.makedirs(self.save_dir, exist_ok=True)

    def _path(self, component):
        suffix = self.COMPONENT_SUFFIXES[component]
        return os.path.join(self.save_dir, f"{self.run_name}{suffix}")

    def save(self, models_dict):
        """Save all model state dicts.

        Args:
            models_dict: dict mapping component name -> nn.Module or None.
                Keys: 'generator', 'vae', 'temporal' or 'regressor', 'discriminator'.
        """
        for name, model in models_dict.items():
            if model is not None and name in self.COMPONENT_SUFFIXES:
                torch.save(model.state_dict(), self._path(name))

    def load(self, models_dict, device="cuda"):
        """Load state dicts into models, stripping module. prefix if needed.

        Args:
            models_dict: dict mapping component name -> nn.Module.
            device: device to map tensors to.
        """
        for name, model in models_dict.items():
            if model is None or name not in self.COMPONENT_SUFFIXES:
                continue
            path = self._path(name)
            if not os.path.exists(path):
                print(f"Warning: checkpoint not found for {name}: {path}")
                continue
            sd = torch.load(path, map_location=device, weights_only=False)
            sd = strip_module_prefix(sd)
            model.load_state_dict(sd)

    def exists(self, component="generator"):
        """Check if a checkpoint exists for a component."""
        return os.path.exists(self._path(component))
