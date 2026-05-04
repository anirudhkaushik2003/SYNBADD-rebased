"""Robustness experiment registry — auto-imports all variants to trigger registration.

When adding a new experiment, drop a file in this directory, decorate the class
with @register_experiment("name"), and add an import below.
"""

# Trigger @register_experiment decorators
import synbadd.experiments.asymmetric_information  # noqa: F401
import synbadd.experiments.reduced_time_input  # noqa: F401
from synbadd.experiments.base import BaseExperiment, ExperimentConfig  # noqa: F401
from synbadd.experiments.registry import (  # noqa: F401
    build_experiment,
    list_experiments,
    register_experiment,
)
