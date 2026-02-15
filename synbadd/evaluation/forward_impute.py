"""Forward impute baseline evaluator."""

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


class ForwardImputeEvaluator(BaseEvaluator):
    """Compare forward imputation baseline against SynBADD.

    Forward imputation uses the most recent previous scan as the
    prediction for the missing timepoint.
    """

    VISIT_ORDER = ["sc", "m06", "m12", "m18", "m24", "m36"]

    def __init__(self, cfg, fold):
        super().__init__(cfg, fold)
        self.df = pd.read_csv(cfg.paths.csv_file)
        self.df["Acq Date"] = pd.to_datetime(self.df["Acq Date"])
        self.df = (
            self.df.sort_values(by=["Subject", "Visit", "Acq Date"])
            .drop_duplicates(subset=["Subject", "Visit"], keep="last")
        )
        # Filter to test fold
        self.df = self.df[self.df[f"fold_{fold}"] == "test"].reset_index(drop=True)

    def compute(self):
        cfg = self.cfg
        fold = self.fold

        synbadd_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{fold}",
            cfg.run_name,
        )
        seg_dir = cfg.paths.masks_dir

        volume_rows = []
        dice_rows = []

        subjects = self.df.groupby("Subject")
        for subject_id, group in subjects:
            for _, row in group.iterrows():
                target_visit = row["Visit"]
                if target_visit == "sc":
                    continue  # No forward impute for screening

                # Find most recent previous scan
                prev_row = self._get_most_recent_scan(subject_id, target_visit)
                if prev_row is None:
                    continue

                target_path = row["File_Path"]
                forward_path = prev_row["File_Path"]

                match = re.search(r"(S\d+_I\d+)", os.path.basename(target_path))
                if not match:
                    continue
                image_id = match.group(1)

                seg_path = find_seg_file(seg_dir, image_id)
                synbadd_path = self._find_synbadd_file(synbadd_dir, image_id)

                if not os.path.exists(forward_path) or seg_path is None:
                    continue

                target_img = load_nii(target_path)
                forward_img = load_nii(forward_path)
                seg_data = load_nii(seg_path)

                for region_name, region_ids in self.roi_regions.items():
                    target_mask = extract_roi(seg_data, region_ids)
                    target_vol = calculate_volume_mm3(target_mask)

                    dice_rows.append({
                        "subject": subject_id,
                        "image_id": image_id,
                        "target_visit": target_visit,
                        "prev_visit": prev_row["Visit"],
                        "region": region_name,
                        "target_volume_mm3": target_vol,
                    })

                ssim_fwd, psnr_fwd = compute_ssim_psnr(target_img, forward_img)
                entry = {
                    "subject": subject_id,
                    "image_id": image_id,
                    "target_visit": target_visit,
                    "prev_visit": prev_row["Visit"],
                    "ssim_forward": ssim_fwd,
                    "psnr_forward": psnr_fwd,
                }
                if synbadd_path is not None:
                    synbadd_img = load_nii(synbadd_path)
                    ssim_syn, psnr_syn = compute_ssim_psnr(target_img, synbadd_img)
                    entry["ssim_synbadd"] = ssim_syn
                    entry["psnr_synbadd"] = psnr_syn
                volume_rows.append(entry)

        return {
            "roi_volumes": pd.DataFrame(volume_rows) if volume_rows else pd.DataFrame(),
            "dice_summary": pd.DataFrame(dice_rows) if dice_rows else pd.DataFrame(),
        }

    def _get_most_recent_scan(self, subject_id, target_visit):
        """Find the most recent scan before target_visit for a subject."""
        target_idx = self.VISIT_ORDER.index(target_visit)
        subject_data = self.df[self.df["Subject"] == subject_id]

        for idx in range(target_idx - 1, -1, -1):
            visit = self.VISIT_ORDER[idx]
            match = subject_data[subject_data["Visit"] == visit]
            if not match.empty:
                return match.iloc[0]
        return None

    @staticmethod
    def _find_synbadd_file(synbadd_dir, image_id):
        if not os.path.isdir(synbadd_dir):
            return None
        for fname in os.listdir(synbadd_dir):
            if image_id in fname and "synthetic.nii.gz" in fname:
                return os.path.join(synbadd_dir, fname)
        return None
