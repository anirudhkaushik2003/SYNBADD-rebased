"""Average impute baseline evaluator."""

import os
import re

import pandas as pd

from synbadd.evaluation.base_evaluator import BaseEvaluator
from synbadd.evaluation.metrics import (
    calculate_volume_mm3,
    compute_ssim_psnr,
    extract_roi,
    find_seg_file,
    load_nii,
)


class AvgImputeEvaluator(BaseEvaluator):
    """Compare average imputation baseline against SynBADD.

    Average imputation uses the temporal mean of all available scans
    for a subject as the prediction for the missing timepoint.
    """

    def compute(self):
        cfg = self.cfg
        fold = self.fold

        avg_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{fold}",
            f"FINAL_Avg_impute_fold_{fold}",
        )
        synbadd_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{fold}",
            cfg.run_name,
        )
        seg_dir = cfg.paths.masks_dir

        volume_rows = []
        dice_rows = []

        if not os.path.isdir(avg_dir):
            print(f"Warning: avg impute directory not found: {avg_dir}")
            return {"roi_volumes_averaged": pd.DataFrame(), "dice_summary_averaged": pd.DataFrame()}

        for fname in sorted(os.listdir(avg_dir)):
            if not fname.endswith("_synthetic.nii.gz"):
                continue

            match = re.search(r"(S\d+_I\d+)", fname)
            if not match:
                continue
            image_id = match.group(1)

            avg_path = os.path.join(avg_dir, fname)
            synbadd_path = self._find_synbadd_file(synbadd_dir, image_id)
            seg_path = find_seg_file(seg_dir, image_id)

            if seg_path is None:
                continue

            avg_img = load_nii(avg_path)
            seg_data = load_nii(seg_path)
            synbadd_img = load_nii(synbadd_path) if synbadd_path else None

            for region_name, region_ids in self.roi_regions.items():
                target_mask = extract_roi(seg_data, region_ids)
                target_vol = calculate_volume_mm3(target_mask)

                dice_rows.append({
                    "image_id": image_id,
                    "region": region_name,
                    "target_volume_mm3": target_vol,
                })

            ssim_avg, psnr_avg = compute_ssim_psnr(seg_data, avg_img)
            row = {
                "image_id": image_id,
                "ssim_averaged": ssim_avg,
                "psnr_averaged": psnr_avg,
            }
            if synbadd_img is not None:
                ssim_syn, psnr_syn = compute_ssim_psnr(seg_data, synbadd_img)
                row["ssim_synbadd"] = ssim_syn
                row["psnr_synbadd"] = psnr_syn
            volume_rows.append(row)

        return {
            "roi_volumes_averaged": pd.DataFrame(volume_rows) if volume_rows else pd.DataFrame(),
            "dice_summary_averaged": pd.DataFrame(dice_rows) if dice_rows else pd.DataFrame(),
        }

    @staticmethod
    def _find_synbadd_file(synbadd_dir, image_id):
        if not os.path.isdir(synbadd_dir):
            return None
        for fname in os.listdir(synbadd_dir):
            if image_id in fname and "synthetic.nii.gz" in fname:
                return os.path.join(synbadd_dir, fname)
        return None
