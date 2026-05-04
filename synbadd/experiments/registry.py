"""Experiment registry: @register_experiment + build_experiment factory."""

from typing import Dict, List, Type

from synbadd.experiments.base import BaseExperiment

_EXPERIMENT_REGISTRY: Dict[str, Type[BaseExperiment]] = {}


def register_experiment(name: str):
    """Decorator to register an experiment class by name."""
    def decorator(cls):
        if name in _EXPERIMENT_REGISTRY:
            raise ValueError(
                f"Experiment '{name}' already registered by {_EXPERIMENT_REGISTRY[name].__name__}"
            )
        cls.name = name
        _EXPERIMENT_REGISTRY[name] = cls
        return cls
    return decorator


def build_experiment(name: str, cfg, fold: int) -> BaseExperiment:
    """Instantiate a registered experiment by name."""
    if name not in _EXPERIMENT_REGISTRY:
        available = ", ".join(sorted(_EXPERIMENT_REGISTRY))
        raise KeyError(f"Experiment '{name}' not found. Available: {available}")
    return _EXPERIMENT_REGISTRY[name](cfg, fold)


def list_experiments() -> List[str]:
    """Return list of all registered experiment names."""
    return sorted(_EXPERIMENT_REGISTRY)
