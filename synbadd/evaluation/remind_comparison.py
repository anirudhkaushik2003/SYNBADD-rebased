"""ReMiND baseline comparison evaluator."""

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


class ReMiNDComparison(BaseEvaluator):
    """Compare SynBADD outputs against ReMiND baseline.

    Includes trajectory alignment analysis for longitudinal volume tracking.
    """

    def compute(self):
        cfg = self.cfg
        fold = self.fold

        remind_dir = cfg.paths.baselines.remind
        synbadd_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{fold}",
            cfg.run_name,
        )
        seg_dir = cfg.paths.masks_dir

        volume_rows = []
        dice_rows = []

        for fname in sorted(os.listdir(remind_dir)):
            if not fname.endswith(".nii.gz"):
                continue

            match = re.search(r"(S\d+_I\d+)", fname)
            if not match:
                continue
            image_id = match.group(1)

            remind_path = os.path.join(remind_dir, fname)
            synbadd_path = self._find_synbadd_file(synbadd_dir, image_id)
            seg_path = find_seg_file(seg_dir, image_id)

            if synbadd_path is None or seg_path is None:
                continue

            remind_img = load_nii(remind_path)
            synbadd_img = load_nii(synbadd_path)
            seg_data = load_nii(seg_path)

            for region_name, region_ids in self.roi_regions.items():
                target_mask = extract_roi(seg_data, region_ids)
                target_vol = calculate_volume_mm3(target_mask)

                dice_rows.append({
                    "image_id": image_id,
                    "region": region_name,
                    "target_volume_mm3": target_vol,
                })

            ssim_remind, psnr_remind = compute_ssim_psnr(seg_data, remind_img)
            ssim_synbadd, psnr_synbadd = compute_ssim_psnr(seg_data, synbadd_img)

            volume_rows.append({
                "image_id": image_id,
                "ssim_remind": ssim_remind,
                "psnr_remind": psnr_remind,
                "ssim_synbadd": ssim_synbadd,
                "psnr_synbadd": psnr_synbadd,
            })

        return {
            "roi_volumes": pd.DataFrame(volume_rows) if volume_rows else pd.DataFrame(),
            "dice_summary": pd.DataFrame(dice_rows) if dice_rows else pd.DataFrame(),
        }

    @staticmethod
    def _find_synbadd_file(synbadd_dir, image_id):
        if not os.path.isdir(synbadd_dir):
            return None
        for fname in os.listdir(synbadd_dir):
            if image_id in fname and "synthetic.nii.gz" in fname:
                return os.path.join(synbadd_dir, fname)
        return None
