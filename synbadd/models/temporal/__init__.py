"""Temporal model registry — auto-imports all variants to trigger registration."""

import synbadd.models.temporal.bidirectional  # noqa: F401

# Import all temporal model modules to trigger @register_model decorators.
# When adding a new temporal model, add an import here.
import synbadd.models.temporal.cross_attention  # noqa: F401
import synbadd.models.temporal.cross_attention_gated  # noqa: F401
import synbadd.models.temporal.gated_cross_attention  # noqa: F401
import synbadd.models.temporal.gated_transformer  # noqa: F401
import synbadd.models.temporal.regressor  # noqa: F401
from synbadd.models.temporal.base import BaseTemporalModel  # noqa: F401
