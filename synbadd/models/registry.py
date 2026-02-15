"""Model registry with decorator-based registration.

Usage:
    @register_model("MyModel")
    class MyModel(nn.Module):
        ...

    model = build_model("MyModel", input_dim=512)
"""

import torch.nn as nn

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered by {_MODEL_REGISTRY[name].__name__}")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return list of all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())
