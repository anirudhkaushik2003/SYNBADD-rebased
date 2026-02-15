"""Pipeline orchestrator: iterate folds x stages with prereq checking and resume."""

import dataclasses
import os
import traceback

from synbadd.pipeline.manifest import ExperimentManifest
from synbadd.pipeline.stages import EvaluateStage, InferStage, TrainStage
from synbadd.utils import get_run_name, seed_everything

# Stage name -> class mapping
_STAGE_CLASSES = {
    "train": TrainStage,
    "infer": InferStage,
    "evaluate": EvaluateStage,
}


class Pipeline:
    """Run train -> infer -> evaluate across multiple folds.

    Args:
        cfg: Base config (fold will be overridden per iteration).
        folds: List of fold indices to run.
        stages: List of stage names to execute (subset of train, infer, evaluate).
        comparison_type: Evaluation comparison type (for evaluate stage).
    """

    def __init__(self, cfg, folds, stages, comparison_type="noder", infer_method="synbadd"):
        self.base_cfg = cfg
        self.folds = folds
        self.stages = stages
        self.comparison_type = comparison_type
        self.infer_method = infer_method

        # Validate stage names
        for s in stages:
            if s not in _STAGE_CLASSES:
                raise ValueError(
                    f"Unknown stage '{s}'. Available: {', '.join(_STAGE_CLASSES)}"
                )

    def run(self, skip_existing=False):
        """Execute the pipeline across all folds and stages.

        Args:
            skip_existing: If True, skip stages whose outputs already exist.
        """
        results = {}

        for fold in self.folds:
            print(f"\n{'='*60}")
            print(f"  FOLD {fold}")
            print(f"{'='*60}")

            # Create fold-specific config
            fold_cfg = dataclasses.replace(
                self.base_cfg,
                data=dataclasses.replace(self.base_cfg.data, fold=fold),
            )

            # Generate run name if not set
            if not fold_cfg.run_name:
                fold_cfg = dataclasses.replace(fold_cfg, run_name=get_run_name(fold_cfg))

            # Seed for reproducibility
            seed_everything(fold_cfg.system.seed)

            # Initialize manifest
            manifest_dir = os.path.join(
                fold_cfg.paths.checkpoint_dir, "manifests"
            )
            manifest = ExperimentManifest(manifest_dir, fold_cfg.run_name)
            manifest.set_config(fold_cfg)

            print(f"  Run name: {fold_cfg.run_name}")
            print(f"  Manifest: {manifest.path}")

            fold_results = {}
            for stage_name in self.stages:
                print(f"\n  --- Stage: {stage_name} ---")

                stage = self._build_stage(stage_name, fold_cfg, fold)

                # Skip if outputs exist and flag is set
                if skip_existing and stage.outputs_exist():
                    print(f"  [SKIP] {stage_name}: outputs already exist")
                    manifest.update_stage(stage_name, "skipped")
                    continue

                # Check prerequisites
                try:
                    stage.check_prerequisites()
                except FileNotFoundError as e:
                    print(f"  [FAIL] {stage_name}: prerequisite missing — {e}")
                    manifest.update_stage(stage_name, "failed", error=str(e))
                    break  # Don't continue to later stages

                # Run stage
                manifest.update_stage(stage_name, "running")
                try:
                    result = stage.run()
                    fold_results[stage_name] = result

                    # Extract summary for manifest
                    summary = self._extract_summary(stage_name, result)
                    manifest.update_stage(stage_name, "completed", **summary)
                    print(f"  [DONE] {stage_name}")

                except Exception as e:
                    print(f"  [FAIL] {stage_name}: {e}")
                    traceback.print_exc()
                    manifest.update_stage(stage_name, "failed", error=str(e))
                    break  # Don't continue to later stages

            results[fold] = fold_results

        print(f"\n{'='*60}")
        print("  Pipeline complete.")
        print(f"{'='*60}")
        return results

    def _build_stage(self, stage_name, cfg, fold):
        """Instantiate a stage object."""
        if stage_name == "evaluate":
            return EvaluateStage(cfg, fold, comparison_type=self.comparison_type)
        if stage_name == "infer":
            return InferStage(cfg, fold, method=self.infer_method)
        cls = _STAGE_CLASSES[stage_name]
        return cls(cfg, fold)

    @staticmethod
    def _extract_summary(stage_name, result):
        """Pull key metrics from stage results into manifest summary."""
        if result is None:
            return {}

        if stage_name == "train" and isinstance(result, dict):
            return {
                "best_val_loss": result.get("best_val_loss"),
                "best_epoch": result.get("best_epoch"),
                "wandb_run_id": result.get("wandb_run_id"),
            }
        elif stage_name == "infer" and isinstance(result, dict):
            return {
                "output_dir": result.get("output_dir"),
                "n_subjects": result.get("n_subjects"),
            }
        elif stage_name == "evaluate" and isinstance(result, dict):
            return {"summary": result}

        return {}
