"""NODER baseline comparison evaluator."""

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


class NODERComparison(BaseEvaluator):
    """Compare SynBADD outputs against NODER baseline."""

    def compute(self):
        cfg = self.cfg
        fold = self.fold

        noder_dir = cfg.paths.baselines.noder
        synbadd_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{fold}",
            cfg.run_name,
        )
        seg_dir = cfg.paths.masks_dir

        # Collect results
        volume_rows = []
        dice_rows = []

        # Iterate NODER output files
        for fname in sorted(os.listdir(noder_dir)):
            if not fname.endswith(".nii.gz"):
                continue

            # Extract image ID
            match = re.search(r"(S\d+_I\d+)", fname)
            if not match:
                continue
            image_id = match.group(1)

            noder_path = os.path.join(noder_dir, fname)

            # Find matching SynBADD and target files
            synbadd_path = self._find_synbadd_file(synbadd_dir, image_id)
            seg_path = find_seg_file(seg_dir, image_id)

            if synbadd_path is None or seg_path is None:
                continue

            # Load volumes
            noder_img = load_nii(noder_path)
            synbadd_img = load_nii(synbadd_path)
            seg_data = load_nii(seg_path)

            # Compute per-ROI metrics
            for region_name, region_ids in self.roi_regions.items():
                target_mask = extract_roi(seg_data, region_ids)
                target_vol = calculate_volume_mm3(target_mask)

                dice_rows.append({
                    "image_id": image_id,
                    "region": region_name,
                    "target_volume_mm3": target_vol,
                })

            # Image-level SSIM/PSNR
            ssim_noder, psnr_noder = compute_ssim_psnr(seg_data, noder_img)
            ssim_synbadd, psnr_synbadd = compute_ssim_psnr(seg_data, synbadd_img)

            volume_rows.append({
                "image_id": image_id,
                "ssim_noder": ssim_noder,
                "psnr_noder": psnr_noder,
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
