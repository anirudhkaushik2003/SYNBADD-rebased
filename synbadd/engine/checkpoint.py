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


def add_module_prefix(state_dict):
    """Add 'module.' prefix for loading into DataParallel-wrapped models."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith("module."):
            new_sd[f"module.{k}"] = v
        else:
            new_sd[k] = v
    return new_sd


def adapt_state_dict(state_dict, model):
    """Automatically adapt state_dict keys to match the model.

    Handles all four combinations:
      - checkpoint has 'module.' prefix, model expects 'module.' (DataParallel) -> keep as-is
      - checkpoint has 'module.' prefix, model expects bare keys -> strip prefix
      - checkpoint has bare keys, model expects 'module.' (DataParallel) -> add prefix
      - checkpoint has bare keys, model expects bare keys -> keep as-is
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())

    # Already compatible
    if model_keys == sd_keys:
        return state_dict

    # Try stripping module. prefix
    stripped = strip_module_prefix(state_dict)
    if set(stripped.keys()) == model_keys:
        return stripped

    # Try adding module. prefix
    prefixed = add_module_prefix(state_dict)
    if set(prefixed.keys()) == model_keys:
        return prefixed

    # Try stripping then adding (handles double module. or other edge cases)
    re_prefixed = add_module_prefix(stripped)
    if set(re_prefixed.keys()) == model_keys:
        return re_prefixed

    # Architecture mismatch: keys differ even after prefix adjustment.
    # Return None to signal the caller to use strict=False loading.
    return None


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
            adapted = adapt_state_dict(sd, model)
            if adapted is None:
                # Architecture mismatch — try adding module. prefix then load with strict=False
                model_has_module = any(k.startswith("module.") for k in model.state_dict())
                sd_has_module = any(k.startswith("module.") for k in sd)
                if model_has_module and not sd_has_module:
                    best_sd = add_module_prefix(sd)
                elif not model_has_module and sd_has_module:
                    best_sd = strip_module_prefix(sd)
                else:
                    best_sd = sd
                missing, unexpected = model.load_state_dict(best_sd, strict=False)
                if missing or unexpected:
                    print(f"Warning: architecture mismatch loading '{name}' checkpoint.")
                    print(f"  Missing  : {missing[:5]}{'...' if len(missing)>5 else ''}")
                    print(f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
                    print("  Proceeding with partial load (strict=False).")
            else:
                model.load_state_dict(adapted)

    def exists(self, component="generator"):
        """Check if a checkpoint exists for a component."""
        return os.path.exists(self._path(component))
