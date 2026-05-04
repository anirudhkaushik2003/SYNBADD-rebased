"""Experiment 2 — Effect of Asymmetric Context Availability.

Fix target = group's midpoint (m12) and mask one input timepoint at a time.
The x-axis for aggregation is the masked timepoint. AD is excluded — without
m36 the asymmetry test reduces to a degenerate 2-config case.

Per-group setup
---------------
CN  -> target m12 ; inputs {m06, m24, m36}        -> 4 configs
MCI -> target m12 ; inputs {m06, m18, m24, m36}   -> 5 configs
AD  -> excluded
"""

from synbadd.experiments.base import BaseExperiment, ExperimentConfig
from synbadd.experiments.registry import register_experiment


_TP_MONTH = {"sc": 0, "m06": 6, "m12": 12, "m18": 18, "m24": 24, "m36": 36}


_GROUP_PARAMS = {
    "CN":  {
        "midpoint": "m12",
        "input_tps": ["m06", "m24", "m36"],
    },
    "MCI": {
        "midpoint": "m12",
        "input_tps": ["m06", "m18", "m24", "m36"],
    },
    # AD intentionally absent — see module docstring.
}


@register_experiment("asymmetric_information")
class AsymmetricInformationExperiment(BaseExperiment):
    """Mask one input timepoint at a time around a fixed midpoint target."""

    aggregation_key = "masked_tp"
    plot_kind = "bar"

    def required_timepoints(self, group):
        params = _GROUP_PARAMS.get(group)
        if params is None:
            return None
        return ["sc", params["midpoint"]] + params["input_tps"]

    def configs_for_subject(self, subject_id, group, available_tps):
        params = _GROUP_PARAMS[group]
        target = params["midpoint"]
        input_tps = params["input_tps"]
        target_month = _TP_MONTH[target]

        configs = [ExperimentConfig(
            config_id="full",
            target_tp=target,
            included_tps=list(input_tps),
            metadata={
                "masked_tp": "none",
                "masked_distance_mo": 0,
                "masked_direction": "none",
            },
        )]

        for mask_tp in input_tps:
            kept = [tp for tp in input_tps if tp != mask_tp]
            mask_month = _TP_MONTH[mask_tp]
            distance = abs(mask_month - target_month)
            direction = "past" if mask_month < target_month else "future"
            configs.append(ExperimentConfig(
                config_id=f"drop_{mask_tp}",
                target_tp=target,
                included_tps=kept,
                metadata={
                    "masked_tp": mask_tp,
                    "masked_distance_mo": distance,
                    "masked_direction": direction,
                },
            ))
        return configs
