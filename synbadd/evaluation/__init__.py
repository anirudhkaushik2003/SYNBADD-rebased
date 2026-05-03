"""Evaluation and comparison analysis."""

from synbadd.evaluation.base_evaluator import BaseEvaluator  # noqa: F401


def build_evaluator(comparison_type: str, cfg, **kwargs):
    """Factory to create the appropriate evaluator.

    Extra kwargs are forwarded to the evaluator constructor.
    For ablation comparisons, pass run_names and variant_labels.
    """
    _EVALUATORS = {
        "noder": "synbadd.evaluation.noder_comparison.NODERComparison",
        "remind": "synbadd.evaluation.remind_comparison.ReMiNDComparison",
        "ablation": "synbadd.evaluation.ablation_comparison.AblationComparison",
        "avg_impute": "synbadd.evaluation.avg_impute.AvgImputeEvaluator",
        "forward_impute": "synbadd.evaluation.forward_impute.ForwardImputeEvaluator",
    }
    if comparison_type not in _EVALUATORS:
        available = ", ".join(sorted(_EVALUATORS.keys()))
        raise KeyError(f"Unknown comparison '{comparison_type}'. Available: {available}")

    module_path, class_name = _EVALUATORS[comparison_type].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(cfg, fold=cfg.data.fold, **kwargs)
