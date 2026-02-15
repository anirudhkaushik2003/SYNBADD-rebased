"""Model registry and factory functions."""

import synbadd.models.discriminator  # noqa: F401
import synbadd.models.generator  # noqa: F401
import synbadd.models.temporal  # noqa: F401

# Import all model modules to trigger registration
import synbadd.models.vae  # noqa: F401
from synbadd.models.registry import build_model, register_model

__all__ = ["register_model", "build_model"]
