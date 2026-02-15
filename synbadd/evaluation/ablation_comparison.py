"""Ablation comparison evaluator: compare multiple SynBADD model variants."""

import os
import re

import pandas as pd

from synbadd.evaluation.base_evaluator import BaseEvaluator
from synbadd.evaluation.metrics import (
    compute_ssim_psnr,
    extract_roi,
    find_seg_file,
    load_nii,
)


class AblationComparison(BaseEvaluator):
    """Compare multiple SynBADD model variants on the same test fold.

    Requires a list of run_names to compare, set via cfg or passed to compute().
    """

    def __init__(self, cfg, fold, run_names=None):
        super().__init__(cfg, fold)
        self.run_names = run_names or []

    def compute(self):
        cfg = self.cfg
        fold = self.fold
        seg_dir = cfg.paths.seg_dir or cfg.paths.masks_dir

        dice_rows = []
        metric_rows = []

        for run_name in self.run_names:
            variant_dir = os.path.join(
                cfg.paths.output_dir,
                f"SYNBADD_Qualitative_{fold}",
                run_name,
            )
            if not os.path.isdir(variant_dir):
                print(f"Warning: directory not found for {run_name}: {variant_dir}")
                continue

            for fname in sorted(os.listdir(variant_dir)):
                if not fname.endswith("_synthetic.nii.gz"):
                    continue

                match = re.search(r"(S\d+_I\d+)", fname)
                if not match:
                    continue
                image_id = match.group(1)

                syn_path = os.path.join(variant_dir, fname)
                seg_path = find_seg_file(seg_dir, image_id)
                if seg_path is None:
                    continue

                syn_img = load_nii(syn_path)
                seg_data = load_nii(seg_path)

                # Per-ROI Dice
                for region_name, region_ids in self.roi_regions.items():
                    extract_roi(seg_data, region_ids)
                    dice_rows.append({
                        "run_name": run_name,
                        "image_id": image_id,
                        "region": region_name,
                    })

                # Image-level SSIM/PSNR
                ssim_val, psnr_val = compute_ssim_psnr(seg_data, syn_img)
                metric_rows.append({
                    "run_name": run_name,
                    "image_id": image_id,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                })

        return {
            "dice_scores": pd.DataFrame(dice_rows) if dice_rows else pd.DataFrame(),
            "per_image_metrics": pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(),
        }
