"""Ablation comparison evaluator: compare multiple SynBADD model variants.

For each synthetic scan:
  1. ANTs SyN registration: synthetic -> target
  2. Warp target segmentation mask to synthetic space
  3. Per-ROI Dice between warped mask and original mask
  4. Per-ROI volume computation
  5. Image-level SSIM/PSNR

Results are logged per-image and as summaries to W&B.
"""

import os
import re

import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from synbadd.evaluation.base_evaluator import BaseEvaluator


def _load_nii(path, target_shape=(128, 128, 128)):
    """Load NIfTI, pad/center to target_shape, normalize to [0,1]."""
    data = nib.load(path).get_fdata()
    if data.ndim == 4:
        data = data.squeeze(0)
    data = data.astype(np.float32)

    if data.shape != target_shape:
        padded = np.zeros(target_shape, dtype=np.float32)
        offsets = [max(0, (t - s) // 2) for s, t in zip(data.shape, target_shape)]
        slices_src = [slice(0, min(s, t)) for s, t in zip(data.shape, target_shape)]
        slices_dst = [slice(o, o + min(s, t)) for o, s, t in zip(offsets, data.shape, target_shape)]
        padded[tuple(slices_dst)] = data[tuple(slices_src)]
        data = padded

    fg = data[data > 0]
    if len(fg) > 0:
        dmin, dmax = fg.min(), fg.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
        data[data < 0] = 0.0
    return data


def _dice(pred, target):
    intersection = np.sum(pred * target)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)


def _extract_roi(seg, region_ids):
    mask = np.zeros_like(seg)
    for rid in region_ids:
        mask[seg == rid] = 1
    return mask


def _volume_mm3(mask, voxel_size=(2, 2, 2)):
    return float(np.sum(mask) * np.prod(voxel_size))


def _find_seg(seg_dir, image_id, all_seg_files):
    """Find DKTatlas segmentation file for an image ID."""
    suffix = "_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz"
    for f in all_seg_files:
        if image_id in f and suffix in f:
            return os.path.join(seg_dir, f)
    # Fallback: any .nii.gz with the image_id
    for f in all_seg_files:
        if image_id in f and f.endswith(".nii.gz"):
            return os.path.join(seg_dir, f)
    return None


class AblationComparison(BaseEvaluator):
    """Compare multiple SynBADD model variants on the same test fold.

    Args:
        cfg: SynbaddConfig
        fold: fold index
        run_names: list of run directory names to compare
        variant_labels: short labels for each variant (for W&B/CSV columns)
    """

    def __init__(self, cfg, fold, run_names=None, variant_labels=None):
        super().__init__(cfg, fold)
        self.run_names = run_names or []
        self.variant_labels = variant_labels or self.run_names

        # Load test CSV
        self.df = pd.read_csv(cfg.paths.csv_file)
        self.df["Acq Date"] = pd.to_datetime(self.df["Acq Date"])
        self.df = (
            self.df.sort_values(by=["Subject", "Visit", "Acq Date"])
            .drop_duplicates(subset=["Subject", "Visit"], keep="last")
        )
        self.df = self.df[self.df[f"fold_{fold}"] == "test"].reset_index(drop=True)

        # Segmentation directory
        self.seg_dir = cfg.paths.seg_dir
        self.all_seg_files = os.listdir(self.seg_dir) if os.path.isdir(self.seg_dir) else []

    def compute(self):
        """Run full comparison: ANTs registration, Dice, volume, SSIM/PSNR."""
        cfg = self.cfg
        fold = self.fold
        regions = self.roi_regions
        base_dir = os.path.join(cfg.paths.output_dir, f"SYNBADD_Qualitative_{fold}")

        # Collect files per variant
        variant_dirs = []
        variant_file_lists = []
        for rn in self.run_names:
            d = os.path.join(base_dir, rn)
            if not os.path.isdir(d):
                print(f"Warning: directory not found: {d}")
                variant_dirs.append(d)
                variant_file_lists.append([])
                continue
            variant_dirs.append(d)
            variant_file_lists.append(os.listdir(d))

        # Use first variant's synthetic files as the iteration base
        if not variant_file_lists or not variant_file_lists[0]:
            print("No synthetic files found for first variant.")
            return {"dice_scores": pd.DataFrame(), "volume_results": pd.DataFrame(),
                    "per_image_metrics": pd.DataFrame(), "summary_metrics": pd.DataFrame()}

        syn_files = sorted([f for f in variant_file_lists[0]
                           if f.endswith("_synthetic.nii.gz") and "_field" not in f])

        n_variants = len(self.run_names)
        labels = self.variant_labels

        # Accumulators
        dice_rows = []
        volume_rows = []
        metric_rows = []

        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        for syn_file in tqdm(syn_files, desc=f"Ablation comparison fold {fold}"):
            match = re.search(r"(S\d+_I\d+)", syn_file)
            if not match:
                continue
            image_id = match.group(1)

            try:
                # Find target scan
                target_rows = self.df[self.df["File_Path"].str.contains(image_id)]
                if target_rows.empty:
                    continue
                target_path = target_rows["File_Path"].values[0]
                subject_id = target_rows["Subject"].values[0]

                if not os.path.exists(target_path):
                    continue

                # Find target segmentation
                seg_path = _find_seg(self.seg_dir, image_id, self.all_seg_files)
                if seg_path is None:
                    continue

                # Load target
                target_img = _load_nii(target_path)
                seg_data = nib.load(seg_path).get_fdata()

                target_ants = ants.from_numpy(target_img)
                seg_ants = ants.from_numpy(seg_data)

                # Process each variant
                variant_imgs = []
                variant_warped_masks = []
                skip = False

                for vi in range(n_variants):
                    # Find synthetic file for this variant
                    var_matches = [f for f in variant_file_lists[vi]
                                   if image_id in f and "_synthetic.nii.gz" in f and "_field" not in f]
                    if not var_matches:
                        skip = True
                        break

                    var_path = os.path.join(variant_dirs[vi], var_matches[0])
                    var_img = _load_nii(var_path)
                    variant_imgs.append(var_img)

                    # ANTs SyN registration: target -> synthetic space
                    var_ants = ants.from_numpy(var_img)
                    reg = ants.registration(fixed=var_ants, moving=target_ants,
                                            type_of_transform="SyN")

                    # Warp segmentation mask
                    warped_mask = ants.apply_transforms(
                        fixed=var_ants, moving=seg_ants,
                        transformlist=reg["fwdtransforms"],
                        interpolator="nearestNeighbor",
                    )
                    variant_warped_masks.append(warped_mask.numpy())

                if skip:
                    continue

                # Per-ROI Dice and volume
                for region_name, region_ids in regions.items():
                    target_roi = _extract_roi(seg_data, region_ids)
                    target_vol = _volume_mm3(target_roi)

                    for vi in range(n_variants):
                        var_roi = _extract_roi(variant_warped_masks[vi], region_ids)
                        d = _dice(var_roi, target_roi)
                        var_vol = _volume_mm3(var_roi)
                        vol_diff_pct = ((var_vol - target_vol) / (target_vol + 1e-6)) * 100

                        dice_rows.append({
                            "image_id": image_id,
                            "subject": subject_id,
                            "region": region_name,
                            "variant": labels[vi],
                            "dice": d,
                        })
                        volume_rows.append({
                            "image_id": image_id,
                            "subject": subject_id,
                            "region": region_name,
                            "variant": labels[vi],
                            "target_volume_mm3": target_vol,
                            "synthetic_volume_mm3": var_vol,
                            "volume_diff_pct": vol_diff_pct,
                        })

                # SSIM / PSNR per variant
                for vi in range(n_variants):
                    ssim_val = ssim_fn(target_img, variant_imgs[vi], data_range=1.0)
                    psnr_val = psnr_fn(target_img, variant_imgs[vi], data_range=1.0)
                    metric_rows.append({
                        "image_id": image_id,
                        "subject": subject_id,
                        "variant": labels[vi],
                        "ssim": ssim_val,
                        "psnr": psnr_val,
                    })

            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                continue

        # Build DataFrames
        dice_df = pd.DataFrame(dice_rows)
        volume_df = pd.DataFrame(volume_rows)
        metrics_df = pd.DataFrame(metric_rows)

        # Summary: avg Dice per region per variant
        summary_rows = []
        if not dice_df.empty:
            for region_name in regions:
                row = {"region": region_name}
                for label in labels:
                    subset = dice_df[(dice_df["region"] == region_name) & (dice_df["variant"] == label)]
                    row[f"dice_{label}"] = subset["dice"].mean() if not subset.empty else 0.0
                summary_rows.append(row)

        # Add avg SSIM/PSNR to summary
        ssim_psnr_row = {"region": "IMAGE_QUALITY"}
        if not metrics_df.empty:
            for label in labels:
                subset = metrics_df[metrics_df["variant"] == label]
                ssim_psnr_row[f"ssim_{label}"] = subset["ssim"].mean()
                ssim_psnr_row[f"psnr_{label}"] = subset["psnr"].mean()
        summary_rows.append(ssim_psnr_row)

        summary_df = pd.DataFrame(summary_rows)

        return {
            "dice_scores": dice_df,
            "volume_results": volume_df,
            "per_image_metrics": metrics_df,
            "summary_metrics": summary_df,
        }

    def log_wandb(self, results=None):
        """Log results to W&B with per-image scalars, tables, and charts."""
        if self.cfg.system.debug or not self.cfg.wandb.enabled:
            return
        if results is None:
            results = self.compute()

        import wandb

        wandb.init(
            project="ADNI_ROI_DiceScores_UnetAblations",
            name=f"SYNBADD_UnetAblations_Fold{self.fold}",
            config={"fold": self.fold, "variants": self.variant_labels},
        )

        dice_df = results["dice_scores"]
        volume_df = results["volume_results"]
        metrics_df = results["per_image_metrics"]
        summary_df = results["summary_metrics"]
        labels = self.variant_labels
        regions = self.roi_regions

        # --- Per-image scalar logging (one wandb.log per image) ---
        if not dice_df.empty:
            image_ids = dice_df["image_id"].unique()
            for image_id in image_ids:
                img_dice = dice_df[dice_df["image_id"] == image_id]
                img_vol = volume_df[volume_df["image_id"] == image_id] if not volume_df.empty else pd.DataFrame()
                img_metrics = metrics_df[metrics_df["image_id"] == image_id] if not metrics_df.empty else pd.DataFrame()

                log_dict = {"image_id": image_id}

                # Per-region Dice per variant
                for _, row in img_dice.iterrows():
                    label = row["variant"]
                    region = row["region"]
                    log_dict[f"dice_{region}_{label}"] = row["dice"]

                # Per-region volume diff per variant
                for _, row in img_vol.iterrows():
                    label = row["variant"]
                    region = row["region"]
                    log_dict[f"vol_diff_pct_{region}_{label}"] = row["volume_diff_pct"]

                # SSIM/PSNR per variant
                for _, row in img_metrics.iterrows():
                    label = row["variant"]
                    log_dict[f"ssim_{label}"] = row["ssim"]
                    log_dict[f"psnr_{label}"] = row["psnr"]

                wandb.log(log_dict)

        # --- Tables ---
        if not dice_df.empty:
            wandb.log({"dice_scores_table": wandb.Table(dataframe=dice_df)})
        if not volume_df.empty:
            wandb.log({"volume_results_table": wandb.Table(dataframe=volume_df)})
        if not metrics_df.empty:
            wandb.log({"per_image_metrics_table": wandb.Table(dataframe=metrics_df)})
        if not summary_df.empty:
            wandb.log({"summary_metrics": wandb.Table(dataframe=summary_df)})

        # --- Summary scalars ---
        if not dice_df.empty:
            for region_name in regions:
                for label in labels:
                    subset = dice_df[(dice_df["region"] == region_name) & (dice_df["variant"] == label)]
                    if not subset.empty:
                        wandb.summary[f"avg_dice_{region_name}_{label}"] = subset["dice"].mean()

        if not metrics_df.empty:
            for label in labels:
                subset = metrics_df[metrics_df["variant"] == label]
                if not subset.empty:
                    wandb.summary[f"avg_ssim_{label}"] = subset["ssim"].mean()
                    wandb.summary[f"avg_psnr_{label}"] = subset["psnr"].mean()

        if not volume_df.empty:
            for region_name in regions:
                for label in labels:
                    subset = volume_df[(volume_df["region"] == region_name) & (volume_df["variant"] == label)]
                    if not subset.empty:
                        wandb.summary[f"avg_vol_diff_pct_{region_name}_{label}"] = subset["volume_diff_pct"].abs().mean()

        # --- Bar chart: Dice per region ---
        if not dice_df.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            region_names = list(regions.keys())
            n_vars = len(labels)
            bar_width = 0.8 / n_vars
            x = np.arange(len(region_names))

            for vi, label in enumerate(labels):
                avgs = []
                for rn in region_names:
                    subset = dice_df[(dice_df["region"] == rn) & (dice_df["variant"] == label)]
                    avgs.append(subset["dice"].mean() if not subset.empty else 0.0)
                ax.bar(x + vi * bar_width, avgs, bar_width, label=label, alpha=0.7)

            ax.set_xlabel("Brain Region")
            ax.set_ylabel("Average Dice Score")
            ax.set_title(f"Dice Score Comparison — Fold {self.fold}")
            ax.set_xticks(x + bar_width * (n_vars - 1) / 2)
            ax.set_xticklabels([r.replace("_", " ").title() for r in region_names], rotation=45, ha="right")
            ax.legend()
            fig.tight_layout()
            chart_path = os.path.join(self.output_dir, f"dice_comparison_fold_{self.fold}.png")
            fig.savefig(chart_path)
            plt.close(fig)
            wandb.log({"dice_score_comparison": wandb.Image(chart_path)})

        wandb.finish()
