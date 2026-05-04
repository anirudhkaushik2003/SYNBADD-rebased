"""Experiment 1 — Effect of Reduced Input Timepoints.

For each disease group, fix target = group's natural endpoint, then drop
input timepoints from nearest-to-target outward. K denotes the number of
input scans kept (including sc). The x-axis for aggregation is `n_dropped`.

Per-group setup
---------------
CN  -> target m36 ; inputs {m06, m12, m24}                    ; K in 1..4
MCI -> target m36 ; inputs {m06, m12, m18, m24}               ; K in 1..5
AD  -> target m24 ; inputs {m06, m12}                          ; K in 1..3
"""

from synbadd.experiments.base import BaseExperiment, ExperimentConfig
from synbadd.experiments.registry import register_experiment


# Per-group endpoint and the full set of timepoints required from a subject
_GROUP_PARAMS = {
    "CN":  {"endpoint": "m36", "all_tps": ["sc", "m06", "m12", "m24", "m36"]},
    "MCI": {"endpoint": "m36", "all_tps": ["sc", "m06", "m12", "m18", "m24", "m36"]},
    "AD":  {"endpoint": "m24", "all_tps": ["sc", "m06", "m12", "m24"]},
}


@register_experiment("reduced_time_input")
class ReducedTimeInputExperiment(BaseExperiment):
    """Drop input timepoints one-at-a-time, nearest-to-target first."""

    aggregation_key = "n_dropped"

    def required_timepoints(self, group):
        params = _GROUP_PARAMS.get(group)
        return params["all_tps"] if params else None

    def configs_for_subject(self, subject_id, group, available_tps):
        params = _GROUP_PARAMS[group]
        target = params["endpoint"]
        all_tps = params["all_tps"]

        # Non-sc, non-target inputs in chronological order (farthest-from-target first).
        non_sc_inputs = [tp for tp in all_tps if tp not in ("sc", target)]

        configs = []
        # n_dropped = 0..len(non_sc_inputs); K = 1 + len(remaining inputs).
        # Drop nearest first => trim from the *end* of non_sc_inputs.
        for n_dropped in range(len(non_sc_inputs) + 1):
            kept = non_sc_inputs[: len(non_sc_inputs) - n_dropped]
            dropped = non_sc_inputs[len(non_sc_inputs) - n_dropped:]
            K = 1 + len(kept)
            configs.append(ExperimentConfig(
                config_id=f"K{K}_drop{n_dropped}",
                target_tp=target,
                included_tps=kept,
                metadata={
                    "K": K,
                    "n_dropped": n_dropped,
                    "dropped_tps": "|".join(dropped) if dropped else "none",
                    "kept_tps": "|".join(["sc"] + kept),
                },
            ))
        return configs
