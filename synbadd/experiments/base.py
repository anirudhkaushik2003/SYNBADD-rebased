"""Base class and config dataclass for robustness experiments.

An experiment defines:
  - Per-disease-group required timepoints (cohort filter)
  - Per-subject list of input configurations to run

Each config produces one synthetic .nii.gz + one row of metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """A single (subject, configuration) to run.

    Attributes:
        config_id: Short string identifying this config (used in filenames).
        target_tp: Target timepoint to predict (e.g., 'm36').
        included_tps: Non-sc timepoints whose SVFs are kept with mask=1.
            All other non-sc, non-target timepoints are zeroed with mask=0.
        metadata: Extra fields written to the metrics CSV (e.g., K, n_dropped).
    """
    config_id: str
    target_tp: str
    included_tps: List[str]
    metadata: Dict = field(default_factory=dict)


class BaseExperiment(ABC):
    """ABC for per-group robustness experiments."""

    name: str = ""  # set by @register_experiment

    # Column in the per-config metrics CSV used as the x-axis when aggregating.
    # Subclasses can override (e.g. 'n_dropped' for Exp 1, 'masked_tp' for Exp 2).
    aggregation_key: str = "config_id"

    # Chart style for the W&B plots: "line" for numeric x-axis, "bar" for categorical.
    plot_kind: str = "line"

    def __init__(self, cfg, fold: int):
        self.cfg = cfg
        self.fold = fold

    @abstractmethod
    def required_timepoints(self, group: str) -> Optional[List[str]]:
        """Return timepoints a subject in this group must have to qualify.

        Return None to exclude the group entirely.
        """

    @abstractmethod
    def configs_for_subject(
        self, subject_id: str, group: str, available_tps: List[str]
    ) -> List[ExperimentConfig]:
        """Return the list of configs to run for this subject."""

    @property
    def output_subdir(self) -> str:
        """Subdirectory name under SYNBADD_Qualitative_{fold}/."""
        return f"experiment_{self.name}_fold_{self.fold}"
