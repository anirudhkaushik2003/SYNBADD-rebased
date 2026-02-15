"""Validation and testing engine for SYNBADD."""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from synbadd.losses.similarity import compute_metrics


class Validator:
    """Runs validation or test loops, computing and logging metrics.

    Tracks overall, per-timepoint, and per-class metrics.
    Saves CSV results and optionally logs to W&B.
    """

    TIMEPOINTS = ["sc", "m06", "m12", "m18", "m24", "m36"]
    GROUP_MAP = {0.0: "CN", 1.0: "MCI", 2.0: "AD"}

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.timepoint_map = {i: tp for i, tp in enumerate(self.TIMEPOINTS)}

    def run(self, generator, vae_model, temporal_model, discriminator,
            dataloader, criterion, phase="val"):
        """Run evaluation loop.

        Args:
            generator, vae_model, temporal_model, discriminator: model components.
            dataloader: DataLoader for val or test set.
            criterion: CompositeLoss instance (with LDDMM).
            phase: "val" or "test".

        Returns:
            Average loss (float).
        """
        generator.eval()
        vae_model.eval()
        temporal_model.eval()
        if discriminator is not None:
            discriminator.eval()

        total_loss = 0.0
        overall_metrics = []
        tp_metrics = {tp: [] for tp in self.TIMEPOINTS}
        cls_metrics = {cls: [] for cls in self.GROUP_MAP.values()}

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset), desc=phase.capitalize()) as pbar:
                for i, batch in enumerate(dataloader):
                    metrics_row = self._eval_batch(
                        batch, generator, vae_model, temporal_model,
                        discriminator, criterion,
                    )
                    total_loss += metrics_row["total_loss"]
                    overall_metrics.append(metrics_row)

                    # Per-timepoint and per-class breakdown
                    for b in range(batch["target_timepoint_index"].size(0)):
                        tp_idx = batch["target_timepoint_index"][b].item()
                        tp = self.timepoint_map.get(tp_idx + 1, "unknown")
                        if tp in tp_metrics:
                            tp_metrics[tp].append(metrics_row)

                        cls_val = batch["class_label"][b].item()
                        cls_name = self.GROUP_MAP.get(cls_val, "unknown")
                        if cls_name in cls_metrics:
                            cls_metrics[cls_name].append(metrics_row)

                    pbar.update(batch["stacked_svf"].size(0))
                    pbar.set_postfix({"loss": metrics_row["total_loss"]})

        avg_loss = total_loss / max(len(dataloader), 1)

        # Save CSV
        self._save_csv(overall_metrics, tp_metrics, cls_metrics, phase)

        # W&B logging
        if not self.cfg.system.debug and self.cfg.wandb.enabled:
            self._log_wandb(overall_metrics, tp_metrics, cls_metrics, phase)

        return avg_loss

    def _eval_batch(self, batch, generator, vae_model, temporal_model,
                    discriminator, criterion):
        """Evaluate a single batch, returning a metrics dict."""
        device = self.device
        stacked_svf = batch["stacked_svf"].to(device).float()
        latest_scan = batch["latest_scan"].to(device).float()
        target_scan = batch["target_scan"].to(device).float()
        class_labels = batch["class_label"].to(device).float()
        reg_labels = batch["reg_label"].to(device).float()
        target_timepoint = batch["target_timepoint"].to(device).float()
        target_svf = batch["target_svf"].to(device).float()

        target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)

        # Averaged SVF imputation
        contextualized_scan = stacked_svf.mean(dim=1)
        y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)

        # Composite loss
        total_loss_val, loss_dict = criterion(
            y_synthetic, target_scan, synthetic_field,
            target_svf=target_svf,
        )

        # Full metrics
        mse, psnr_val, ssim_val, mean_jac, std_jac, _, nrmse, mssim = compute_metrics(
            target_scan, y_synthetic, synthetic_field,
            criterion.lddmm_loss, device,
        )

        return {
            "total_loss": total_loss_val.item(),
            "mse": mse,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "mssim": mssim,
            "mean_jac": mean_jac,
            "std_jac": std_jac,
            "nrmse": nrmse,
            **loss_dict,
        }

    def _save_csv(self, overall, tp_metrics, cls_metrics, phase):
        """Save metrics to CSV."""
        save_dir = os.path.join(
            self.cfg.paths.checkpoint_dir, self.cfg.project_name, phase
        )
        os.makedirs(save_dir, exist_ok=True)

        # Overall averages
        avg = self._avg_metrics(overall, prefix=f"{phase.capitalize()}")
        # Add per-timepoint
        for tp, rows in tp_metrics.items():
            if rows:
                avg.update(self._avg_metrics(rows, prefix=f"{phase.capitalize()} {tp}"))
        # Add per-class
        for cls, rows in cls_metrics.items():
            if rows:
                avg.update(self._avg_metrics(rows, prefix=f"{phase.capitalize()} {cls}"))

        df = pd.DataFrame([avg])
        csv_path = os.path.join(save_dir, f"{self.cfg.run_name}_{phase}.csv")
        df.to_csv(csv_path, index=False)
        print(f"{phase.capitalize()} metrics saved to {csv_path}")

    def _log_wandb(self, overall, tp_metrics, cls_metrics, phase):
        """Log metrics to W&B."""
        try:
            import wandb
        except ImportError:
            return

        avg = self._avg_metrics(overall, prefix=f"{phase.capitalize()}")
        wandb.log(avg)

    @staticmethod
    def _avg_metrics(rows, prefix=""):
        """Average a list of metric dicts."""
        if not rows:
            return {}
        keys = rows[0].keys()
        avg = {}
        for k in keys:
            vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
            if vals:
                label = f"{prefix} {k}" if prefix else k
                avg[label] = np.mean(vals)
        return avg
