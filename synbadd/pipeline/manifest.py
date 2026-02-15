"""Experiment manifest for tracking runs, checkpoints, and metrics."""

import json
import os
from dataclasses import asdict
from datetime import datetime


class ExperimentManifest:
    """JSON manifest linking config -> checkpoints -> metrics -> W&B run.

    Stored at {checkpoint_dir}/manifests/{run_name}.json.
    """

    def __init__(self, manifest_dir, run_name):
        self.manifest_dir = manifest_dir
        self.run_name = run_name
        self.path = os.path.join(manifest_dir, f"{run_name}.json")
        os.makedirs(manifest_dir, exist_ok=True)

        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "run_name": run_name,
                "created_at": datetime.now().isoformat(),
                "config": {},
                "stages": {},
                "fold": None,
            }

    def set_config(self, cfg):
        """Store the full frozen config."""
        self.data["config"] = asdict(cfg)
        self.data["fold"] = cfg.data.fold
        self._save()

    def update_stage(self, stage_name, status, **kwargs):
        """Update a pipeline stage's status and metadata.

        Args:
            stage_name: 'train', 'infer', or 'evaluate'.
            status: 'running', 'completed', or 'failed'.
            **kwargs: Additional metadata (e.g., best_val_loss, output_dir).
        """
        self.data["stages"][stage_name] = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            **kwargs,
        }
        self._save()

    def get_stage_status(self, stage_name):
        """Get the status of a stage, or None if not started."""
        stage = self.data["stages"].get(stage_name, {})
        return stage.get("status")

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)


class ManifestStore:
    """Query across all experiment manifests in a directory."""

    def __init__(self, manifest_dir):
        self.manifest_dir = manifest_dir

    def list_runs(self, fold=None):
        """List all run names, optionally filtered by fold."""
        runs = []
        if not os.path.isdir(self.manifest_dir):
            return runs
        for fname in sorted(os.listdir(self.manifest_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.manifest_dir, fname)) as f:
                data = json.load(f)
            if fold is not None and data.get("fold") != fold:
                continue
            runs.append(data)
        return runs

    def query(self, fold=None, metric=None, top=5):
        """Query runs sorted by a metric.

        Args:
            fold: Filter by fold number.
            metric: Metric name to sort by (searches in stage metadata).
            top: Number of top results to return.

        Returns:
            List of (run_name, metric_value) tuples, sorted descending.
        """
        runs = self.list_runs(fold=fold)
        results = []
        for run in runs:
            val = self._extract_metric(run, metric)
            if val is not None:
                results.append((run["run_name"], val))

        # Sort: lower is better for losses, higher is better for SSIM/PSNR/Dice
        ascending = metric and any(
            kw in metric.lower()
            for kw in ["loss", "mse", "nrmse", "epe", "error"]
        )
        results.sort(key=lambda x: x[1], reverse=not ascending)
        return results[:top]

    @staticmethod
    def _extract_metric(run_data, metric):
        """Search for a metric value in stage metadata."""
        if not metric:
            return None
        for stage_name, stage_data in run_data.get("stages", {}).items():
            if metric in stage_data:
                return stage_data[metric]
            summary = stage_data.get("summary", {})
            if metric in summary:
                return summary[metric]
        return None
