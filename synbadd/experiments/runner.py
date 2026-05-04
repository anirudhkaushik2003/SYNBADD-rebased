"""ExperimentRunner: execute a registered experiment with a pre-trained model.

Workflow:
  1. inference: per (subject, config) -> build custom batch, forward, save .nii.gz
                + capture image metrics in a per-config CSV.
  1.5. dice_eval (optional): per saved synthetic, ANTs SyN-register the target
                seg into synthetic space and compute Dice + volume per ROI.
  2. aggregation: per-group / per-(group, age_bin) summary CSVs + charts + W&B.
  3. cleanup (optional): delete the .nii.gz files, keep the CSVs.
"""

import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from voxelmorph.torch.networks import VxmDense

from synbadd.engine.checkpoint import CheckpointManager, adapt_state_dict
from synbadd.experiments.base import BaseExperiment
from synbadd.losses.lddmm import VxmLDDMMLoss
from synbadd.losses.similarity import compute_metrics
from synbadd.models.registry import build_model

GROUP_MAP = {"CN": 0, "MCI": 1, "AD": 2}
AGE_THRESHOLD = 70  # binary bin: <70 vs >=70

_IMAGE_ID_RE = re.compile(r"(S\d+_I\d+)")


def _compute_epe(target_svf, synthetic_field):
    """L2 end-point error between target and predicted SVF."""
    diff = target_svf - synthetic_field
    return torch.sqrt((diff ** 2).sum(dim=1)).mean().item()


def _age_bin(age: float) -> str:
    return "lt70" if age < AGE_THRESHOLD else "ge70"


class ExperimentRunner:
    """Run a registered experiment on a pre-trained model.

    Args:
        cfg: SynbaddConfig (must have run_name + project_name set).
        experiment: BaseExperiment instance.
        dataset: BaseSVFDataset for the test fold (provides SVF map + scan loader).
    """

    def __init__(self, cfg, experiment: BaseExperiment, dataset):
        self.cfg = cfg
        self.experiment = experiment
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{cfg.data.fold}",
            experiment.output_subdir,
            cfg.run_name,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.per_config_csv = os.path.join(self.output_dir, "per_config_metrics.csv")
        self._models_built = False

    # ------------------------------------------------------------------
    # Model building (mirrors Inferrer._build_models + checkpoint load)
    # ------------------------------------------------------------------

    def _ensure_models(self):
        if self._models_built:
            return
        cfg = self.cfg
        image_size = (cfg.model.img_height,) * 3

        # Generator
        self.generator = build_model(
            cfg.model.generator.type,
            inshape=image_size,
            latent_input_dim=cfg.model.latent_dim,
            c_dim=cfg.model.c_dim,
            use_probs=False,
            int_downsize=cfg.model.generator.int_downsize,
            in_channels=cfg.model.in_channels,
            int_steps=cfg.model.generator.int_steps,
            nb_unet_features=cfg.model.generator.nb_unet_features,
            using_svf=cfg.model.generator.use_deformation_field,
            using_daft=cfg.model.generator.using_daft,
            using_unet=cfg.model.generator.using_unet,
        )

        # Frozen VAE
        self.vae_model = build_model("VAE3D_UNetSkip")
        sd = torch.load(cfg.paths.vae_checkpoint, map_location="cpu", weights_only=False)
        sd = adapt_state_dict(sd, self.vae_model)
        self.vae_model.load_state_dict(sd)
        for p in self.vae_model.parameters():
            p.requires_grad = False

        # Temporal model
        self.temporal_model = build_model(
            cfg.model.temporal.type,
            input_dim=cfg.model.latent_dim,
            num_heads=cfg.model.temporal.num_heads,
            num_layers=cfg.model.temporal.num_layers,
            max_seq_len=cfg.model.temporal.max_seq_len,
            use_positional_embeddings=cfg.model.temporal.use_positional_embeddings,
            use_bidirectional_embeddings=cfg.model.temporal.use_bidirectional_embeddings,
        )

        # DataParallel
        if cfg.system.n_gpu > 1:
            self.generator = nn.DataParallel(self.generator)
            self.vae_model = nn.DataParallel(self.vae_model)
            self.temporal_model = nn.DataParallel(self.temporal_model)

        # Move to device
        self.generator = self.generator.to(self.device)
        self.vae_model = self.vae_model.to(self.device)
        self.temporal_model = self.temporal_model.to(self.device)

        # Load pretrained weights
        ckpt = CheckpointManager(cfg.paths.checkpoint_dir, cfg.project_name, cfg.run_name)
        ckpt.load(
            {"generator": self.generator, "temporal": self.temporal_model},
            device=self.device,
        )

        # LDDMM loss (used by compute_metrics for Jacobian stats)
        vxm_model = VxmDense(
            inshape=image_size,
            nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32]],
        ).to(self.device)
        vsd = torch.load(cfg.paths.vxm_checkpoint, map_location=self.device, weights_only=False)
        vsd = adapt_state_dict(vsd, vxm_model)
        vxm_model.load_state_dict(vsd)
        vxm_model.eval()
        for p in vxm_model.parameters():
            p.requires_grad = False
        if cfg.system.n_gpu > 1:
            vxm_model = nn.DataParallel(vxm_model)
        self.lddmm_loss = VxmLDDMMLoss(vxm_model)

        # Eval modes
        self.generator.eval()
        self.vae_model.eval()
        self.temporal_model.eval()

        self._models_built = True

    def _get_vae_module(self):
        return self.vae_model.module if hasattr(self.vae_model, "module") else self.vae_model

    # ------------------------------------------------------------------
    # Cohort + batch construction
    # ------------------------------------------------------------------

    def _build_cohort(self):
        """Filter test subjects by per-group required-timepoint constraints.

        Also prints per-group counts of (in-fold / qualified-for-experiment)
        so it's obvious why subjects might have been excluded.
        """
        df = self.dataset.df
        sc_meta = (
            df[df["Visit"] == "sc"].drop_duplicates("Subject")
            .set_index("Subject")[["Group", "Age"]]
        )
        visits_per_subj = df.groupby("Subject")["Visit"].apply(set)

        per_group_total = {}
        per_group_kept = {}
        cohort = []
        for subject_id in self.dataset.subjects:
            if subject_id not in sc_meta.index:
                continue
            group = str(sc_meta.loc[subject_id, "Group"])
            age = float(sc_meta.loc[subject_id, "Age"])
            per_group_total[group] = per_group_total.get(group, 0) + 1

            required = self.experiment.required_timepoints(group)
            if required is None:
                continue
            avail = visits_per_subj.loc[subject_id]
            if not set(required).issubset(avail):
                continue

            per_group_kept[group] = per_group_kept.get(group, 0) + 1
            cohort.append((subject_id, group, age, sorted(avail)))

        print(f"  cohort breakdown for fold {self.cfg.data.fold}:")
        for grp in sorted(set(per_group_total) | set(per_group_kept)):
            req = self.experiment.required_timepoints(grp)
            req_str = ",".join(req) if req else "(excluded)"
            print(f"    {grp:<5} test={per_group_total.get(grp, 0):<3} "
                  f"qualified={per_group_kept.get(grp, 0):<3}  required={req_str}")
        print(f"  cohort total = {len(cohort)} subjects")
        return cohort

    def _build_batch(self, subject_id, config, group, age):
        """Build a single-element batch dict for one (subject, config)."""
        ds = self.dataset
        subject_data = ds.subject_groups.loc[subject_id]

        latest_scan = ds.load_scan(subject_id, visit="sc").float()
        target_scan = ds.load_scan(subject_id, visit=config.target_tp).float()
        target_svf = ds._load_target_svf(subject_id, config.target_tp, subject_data)

        timepoints = ds.timepoints
        target_tp_index = timepoints.index(config.target_tp) - 1

        svfs, mask = [], []
        for tp in timepoints:
            if tp == "sc":
                continue
            if tp == config.target_tp or tp not in config.included_tps:
                svfs.append(np.zeros(ds.VOLUME_SHAPE))
                mask.append(0)
                continue
            svf = ds._load_svf(subject_id, tp, subject_data)
            if svf is not None and svf.shape == ds.VOLUME_SHAPE:
                svfs.append(svf)
                mask.append(1)
            else:
                svfs.append(np.zeros(ds.VOLUME_SHAPE))
                mask.append(0)

        target_path = subject_data[subject_data["Visit"] == config.target_tp].iloc[0]["File_Path"]
        target_uid = ds.file_path_to_id[target_path]

        return {
            "stacked_svf": torch.from_numpy(np.stack(svfs)).float().unsqueeze(0),
            "latest_scan": latest_scan.unsqueeze(0),
            "target_scan": target_scan.unsqueeze(0),
            "class_label": torch.tensor([[GROUP_MAP[group]]], dtype=torch.float32),
            "reg_label": torch.tensor([[age]], dtype=torch.float32),
            "target_timepoint": torch.tensor(
                [[ds.timepoint_map[config.target_tp]]], dtype=torch.float32
            ),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            "target_timepoint_index": torch.tensor([[target_tp_index]], dtype=torch.long),
            "target_svf": torch.from_numpy(np.asarray(target_svf)).float().unsqueeze(0),
            "target_uid": torch.tensor([[target_uid]], dtype=torch.int),
            "target_path": target_path,
        }

    # ------------------------------------------------------------------
    # Forward pass (mirrors Inferrer._contextualize_synbadd)
    # ------------------------------------------------------------------

    def _contextualize(self, stacked_svf, target_timepoint_index, latest_scan):
        cfg = self.cfg
        if not (cfg.data.use_past_data or cfg.data.use_future_data):
            return latest_scan
        batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
        stacked_flat = stacked_svf.view(-1, c_dim, h, w, d)

        vae = self._get_vae_module()
        mu, _, skips = vae.encode(stacked_flat)
        for si in range(len(skips)):
            s_shape = skips[si].shape
            skips[si] = skips[si].view(batch_size, temporal_dim, *s_shape[1:]).mean(dim=1)

        latent = mu.view(batch_size, temporal_dim, -1)
        contextualized = self.temporal_model(latent, target_timepoint_index)
        return vae.decode(contextualized, skips)

    # ------------------------------------------------------------------
    # Phase 1: inference
    # ------------------------------------------------------------------

    def run_inference(self, skip_existing=False):
        """For each (subject, config), build batch, run forward, save .nii.gz + metrics row."""
        self._ensure_models()
        cohort = self._build_cohort()
        if not cohort:
            print(f"[{self.experiment.name}] cohort is empty — nothing to run.")
            # Write an empty CSV so aggregation has a file to read.
            pd.DataFrame().to_csv(self.per_config_csv, index=False)
            return {"per_config_csv": self.per_config_csv, "n_inferences": 0}

        # Pre-enumerate all (subject, config) pairs for accurate progress bar.
        plan = []
        for subject_id, group, age, avail in cohort:
            for config in self.experiment.configs_for_subject(subject_id, group, avail):
                plan.append((subject_id, group, age, config))
        print(f"  plan: {len(plan)} (subject, config) inferences")

        rows = []
        with torch.no_grad():
            for subject_id, group, age, config in tqdm(
                plan, desc=f"[{self.experiment.name}] infer", unit="cfg"
            ):
                base = f"{subject_id}_{config.config_id}"
                syn_path = os.path.join(self.output_dir, f"{base}_synthetic.nii.gz")
                field_path = os.path.join(self.output_dir, f"{base}_synthetic_field.nii.gz")

                if skip_existing and os.path.exists(syn_path):
                    continue

                batch = self._build_batch(subject_id, config, group, age)

                stacked_svf = batch["stacked_svf"].to(self.device).float()
                latest_scan = batch["latest_scan"].to(self.device).float()
                target_scan = batch["target_scan"].to(self.device).float()
                target_svf = batch["target_svf"].to(self.device).float()
                class_labels = batch["class_label"].to(self.device).float()
                reg_labels = batch["reg_label"].to(self.device).float()
                target_timepoint = batch["target_timepoint"].to(self.device).float()
                target_timepoint_index = batch["target_timepoint_index"].to(self.device)

                target_labels = torch.cat(
                    [class_labels, reg_labels, target_timepoint], dim=1
                )

                contextualized_scan = self._contextualize(
                    stacked_svf, target_timepoint_index, latest_scan
                )
                y_synthetic, synthetic_field = self.generator(
                    latest_scan, contextualized_scan, target_labels
                )

                # Save outputs
                nib.save(nib.Nifti1Image(y_synthetic[0].cpu().numpy(), None), syn_path)
                nib.save(nib.Nifti1Image(synthetic_field[0].cpu().numpy(), None), field_path)

                # Image metrics
                epe = _compute_epe(target_svf, synthetic_field)
                mse, psnr_v, ssim_v, mean_jac, std_jac, _, nrmse, mssim = compute_metrics(
                    target_scan, y_synthetic, synthetic_field, self.lddmm_loss, self.device
                )

                rows.append({
                    "experiment": self.experiment.name,
                    "subject": subject_id,
                    "group": group,
                    "age": age,
                    "age_bin": _age_bin(age),
                    "config_id": config.config_id,
                    "target_tp": config.target_tp,
                    "included_tps": "|".join(config.included_tps) if config.included_tps else "none",
                    **config.metadata,
                    "ssim": ssim_v,
                    "psnr": psnr_v,
                    "mse": mse,
                    "nrmse": nrmse,
                    "epe": epe,
                    "mean_jac": mean_jac,
                    "std_jac": std_jac,
                    "synthetic_path": syn_path,
                    "target_path": batch["target_path"],
                })

        # Append to existing CSV if skip_existing was used; otherwise overwrite.
        if skip_existing and os.path.exists(self.per_config_csv) and rows:
            old = pd.read_csv(self.per_config_csv)
            df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        else:
            df = pd.DataFrame(rows)
        df.to_csv(self.per_config_csv, index=False)
        return {"per_config_csv": self.per_config_csv, "n_inferences": len(rows)}

    # ------------------------------------------------------------------
    # Phase 1b: recovery — recompute metrics from already-saved .nii.gz files
    # ------------------------------------------------------------------

    def recover_metrics(self):
        """Recompute the per-config metrics CSV from saved .nii.gz outputs.

        Use this when an earlier run saved the synthetic + field .nii.gz files
        but didn't write (or lost) the per_config_metrics.csv. No model loading
        — pure I/O + metric computation.
        """
        cohort = self._build_cohort()
        if not cohort:
            print(f"[{self.experiment.name}] cohort is empty — nothing to recover.")
            pd.DataFrame().to_csv(self.per_config_csv, index=False)
            return {"per_config_csv": self.per_config_csv, "n_recovered": 0, "n_missing": 0}

        plan = []
        for subject_id, group, age, avail in cohort:
            for config in self.experiment.configs_for_subject(subject_id, group, avail):
                plan.append((subject_id, group, age, config))
        print(f"  plan: {len(plan)} (subject, config) pairs to recover")

        device = torch.device("cpu")  # image metrics are cheap on CPU
        rows = []
        n_missing = 0
        n_failed = 0

        for subject_id, group, age, config in tqdm(
            plan, desc=f"[{self.experiment.name}] recover", unit="cfg"
        ):
            base = f"{subject_id}_{config.config_id}"
            syn_path = os.path.join(self.output_dir, f"{base}_synthetic.nii.gz")
            field_path = os.path.join(self.output_dir, f"{base}_synthetic_field.nii.gz")

            if not os.path.exists(syn_path) or not os.path.exists(field_path):
                n_missing += 1
                continue

            try:
                y_arr = nib.load(syn_path).get_fdata()        # [1, H, W, D]
                field_arr = nib.load(field_path).get_fdata()  # [3, H, W, D]
                y_synthetic = torch.from_numpy(y_arr).float().unsqueeze(0)        # [1, 1, H, W, D]
                synthetic_field = torch.from_numpy(field_arr).float().unsqueeze(0)  # [1, 3, H, W, D]

                ds = self.dataset
                subject_data = ds.subject_groups.loc[subject_id]
                target_scan = ds.load_scan(subject_id, visit=config.target_tp).float().unsqueeze(0)
                target_svf = ds._load_target_svf(subject_id, config.target_tp, subject_data)
                target_svf_t = torch.from_numpy(np.asarray(target_svf)).float().unsqueeze(0)
                target_path = subject_data[
                    subject_data["Visit"] == config.target_tp
                ].iloc[0]["File_Path"]

                # compute_metrics ignores its lddmm arg, so passing None is fine
                epe = _compute_epe(target_svf_t, synthetic_field)
                mse, psnr_v, ssim_v, mean_jac, std_jac, _, nrmse, _ = compute_metrics(
                    target_scan, y_synthetic, synthetic_field, None, device
                )
            except Exception as e:
                print(f"    [WARN] failed to recover {base}: {e}")
                n_failed += 1
                continue

            rows.append({
                "experiment": self.experiment.name,
                "subject": subject_id,
                "group": group,
                "age": age,
                "age_bin": _age_bin(age),
                "config_id": config.config_id,
                "target_tp": config.target_tp,
                "included_tps": "|".join(config.included_tps) if config.included_tps else "none",
                **config.metadata,
                "ssim": ssim_v,
                "psnr": psnr_v,
                "mse": mse,
                "nrmse": nrmse,
                "epe": epe,
                "mean_jac": mean_jac,
                "std_jac": std_jac,
                "synthetic_path": syn_path,
                "target_path": target_path,
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.per_config_csv, index=False)
        print(f"  recovered {len(rows)} rows; {n_missing} missing on disk; {n_failed} failed")
        return {
            "per_config_csv": self.per_config_csv,
            "n_recovered": len(rows),
            "n_missing": n_missing,
            "n_failed": n_failed,
        }

    # ------------------------------------------------------------------
    # Phase 1.5: Dice + volume from saved .nii.gz files
    # ------------------------------------------------------------------

    @property
    def per_dice_csv(self):
        return os.path.join(self.output_dir, "per_config_dice_volume.csv")

    def run_dice_eval(self):
        """Compute Dice + volume per (subject, config, ROI) from saved .nii.gz.

        Uses ANTs SyN to register the target scan into synthetic space, warps the
        target's segmentation, and computes per-ROI Dice + volume difference.
        Reads the per-config CSV (built by run_inference / recover_metrics) for
        the (subject, config, paths) plan.
        """
        if not os.path.exists(self.per_config_csv):
            print(f"[{self.experiment.name}] cannot run Dice — per-config CSV missing.")
            return None
        if os.path.getsize(self.per_config_csv) == 0:
            print(f"[{self.experiment.name}] cannot run Dice — per-config CSV empty.")
            return None

        try:
            import ants  # noqa: F401
        except ImportError:
            print(
                f"[{self.experiment.name}] ANTs not installed; skipping Dice/volume.\n"
                "  -> pip install antspyx"
            )
            return None

        from synbadd.evaluation.ablation_comparison import (
            _dice, _extract_roi, _find_seg, _load_nii, _volume_mm3,
        )
        import ants

        seg_dir = self.cfg.paths.seg_dir
        if not os.path.isdir(seg_dir):
            print(f"[{self.experiment.name}] seg_dir not found: {seg_dir}; skipping Dice.")
            return None
        all_seg_files = os.listdir(seg_dir)
        roi_regions = self.cfg.eval.roi_regions
        agg_key = self.experiment.aggregation_key

        df = pd.read_csv(self.per_config_csv)

        # Idempotency: if a complete dice CSV already exists, skip.
        if os.path.exists(self.per_dice_csv) and os.path.getsize(self.per_dice_csv) > 0:
            try:
                existing = pd.read_csv(self.per_dice_csv)
                expected_rows = len(df) * len(roi_regions)
                if len(existing) >= expected_rows:
                    print(f"  Dice CSV already complete ({len(existing)} rows) — skipping. "
                          f"Delete {self.per_dice_csv} to force recompute.")
                    return self.per_dice_csv
                # Partial CSV — recompute only missing (subject, config_id) pairs.
                done = set(zip(existing["subject"], existing["config_id"]))
                df = df[~df.apply(
                    lambda r: (r["subject"], r["config_id"]) in done, axis=1
                )].reset_index(drop=True)
                print(f"  Dice CSV partial; recomputing {len(df)} missing configs.")
            except Exception:
                pass

        if df.empty:
            return self.per_dice_csv
        print(f"  running Dice over {len(df)} (subject, config) rows × "
              f"{len(roi_regions)} ROIs")

        rows = []
        n_skipped = 0
        n_failed = 0

        for _, src_row in tqdm(df.iterrows(), total=len(df),
                               desc=f"[{self.experiment.name}] dice", unit="cfg"):
            syn_path = src_row.get("synthetic_path", "")
            target_path = src_row.get("target_path", "")
            if not (isinstance(syn_path, str) and os.path.exists(syn_path)
                    and isinstance(target_path, str) and os.path.exists(target_path)):
                n_skipped += 1
                continue
            m = _IMAGE_ID_RE.search(os.path.basename(target_path))
            if m is None:
                n_skipped += 1
                continue
            seg_path = _find_seg(seg_dir, m.group(1), all_seg_files)
            if seg_path is None:
                n_skipped += 1
                continue

            try:
                target_img = _load_nii(target_path)
                syn_img = _load_nii(syn_path)
                seg_data = nib.load(seg_path).get_fdata()

                target_ants = ants.from_numpy(target_img)
                syn_ants = ants.from_numpy(syn_img)
                seg_ants = ants.from_numpy(seg_data)

                # Register target -> synthetic space, then warp seg with same xform.
                reg = ants.registration(
                    fixed=syn_ants, moving=target_ants, type_of_transform="SyN"
                )
                warped_mask = ants.apply_transforms(
                    fixed=syn_ants, moving=seg_ants,
                    transformlist=reg["fwdtransforms"],
                    interpolator="nearestNeighbor",
                ).numpy()

                for region_name, region_ids in roi_regions.items():
                    target_roi = _extract_roi(seg_data, region_ids)
                    warped_roi = _extract_roi(warped_mask, region_ids)
                    d = _dice(warped_roi, target_roi)
                    target_vol = _volume_mm3(target_roi)
                    warped_vol = _volume_mm3(warped_roi)
                    vol_diff_pct = (
                        (warped_vol - target_vol) / (target_vol + 1e-6)
                    ) * 100
                    rows.append({
                        "experiment": src_row["experiment"],
                        "subject": src_row["subject"],
                        "group": src_row["group"],
                        "age": src_row["age"],
                        "age_bin": src_row["age_bin"],
                        "config_id": src_row["config_id"],
                        "target_tp": src_row["target_tp"],
                        agg_key: src_row[agg_key],
                        "region": region_name,
                        "dice": d,
                        "target_volume_mm3": target_vol,
                        "synthetic_volume_mm3": warped_vol,
                        "vol_diff_pct": vol_diff_pct,
                    })
            except Exception as e:
                print(f"  [WARN] Dice failed for {src_row['subject']}/"
                      f"{src_row['config_id']}: {e}")
                n_failed += 1
                continue

        new = pd.DataFrame(rows)
        if os.path.exists(self.per_dice_csv) and os.path.getsize(self.per_dice_csv) > 0:
            try:
                existing = pd.read_csv(self.per_dice_csv)
                out = pd.concat([existing, new], ignore_index=True)
            except Exception:
                out = new
        else:
            out = new
        out.to_csv(self.per_dice_csv, index=False)
        print(f"  saved {self.per_dice_csv} (+{len(new)} new rows, "
              f"{len(out)} total; skipped {n_skipped}, failed {n_failed})")
        return self.per_dice_csv

    # ------------------------------------------------------------------
    # Phase 2: aggregation
    # ------------------------------------------------------------------

    def run_aggregation(self):
        """Read per-config CSV, write per-group and per-(group, age_bin) summaries."""
        if not os.path.exists(self.per_config_csv):
            print(
                f"[{self.experiment.name}] per-config CSV missing: {self.per_config_csv}\n"
                "  -> the inference phase did not write a CSV. Likely causes:\n"
                "     - cohort was empty (check the per-group breakdown printed above)\n"
                "     - the inference loop crashed before writing (re-run without --skip-inference)"
            )
            return {}

        if os.path.getsize(self.per_config_csv) == 0:
            print(f"[{self.experiment.name}] per-config CSV is empty — nothing to aggregate.")
            return {}
        df = pd.read_csv(self.per_config_csv)
        if df.empty:
            print(f"[{self.experiment.name}] per-config CSV is empty — nothing to aggregate.")
            return {}

        metric_cols = [c for c in ["ssim", "psnr", "mse", "nrmse", "epe", "mean_jac", "std_jac"]
                       if c in df.columns]
        agg_key = self.experiment.aggregation_key

        def _agg(keys):
            g = df.groupby(keys)[metric_cols].agg(["mean", "std", "count"])
            g.columns = [f"{m}_{s}" for m, s in g.columns]
            return g.reset_index()

        out = {
            "per_group_config": _agg(["group", agg_key]),
            "per_group_agebin_config": _agg(["group", "age_bin", agg_key]),
            "overall_per_config": _agg([agg_key]),
        }

        for name, agg_df in out.items():
            path = os.path.join(self.output_dir, f"agg_{name}.csv")
            agg_df.to_csv(path, index=False)
            print(f"  saved {path} ({len(agg_df)} rows)")

        # Dice/volume aggregation, if a per_config_dice_volume.csv exists.
        dice_df = None
        if os.path.exists(self.per_dice_csv) and os.path.getsize(self.per_dice_csv) > 0:
            dice_df = pd.read_csv(self.per_dice_csv)
            if not dice_df.empty:
                dice_metric_cols = [c for c in ["dice", "vol_diff_pct"] if c in dice_df.columns]

                def _agg_d(keys):
                    g = dice_df.groupby(keys)[dice_metric_cols].agg(["mean", "std", "count"])
                    g.columns = [f"{m}_{s}" for m, s in g.columns]
                    return g.reset_index()

                dice_aggs = {
                    "dice_per_group_region_config": _agg_d(["group", "region", agg_key]),
                    "dice_per_group_agebin_region_config": _agg_d(
                        ["group", "age_bin", "region", agg_key]
                    ),
                    "dice_overall_region_config": _agg_d(["region", agg_key]),
                }
                for name, agg_df in dice_aggs.items():
                    path = os.path.join(self.output_dir, f"agg_{name}.csv")
                    agg_df.to_csv(path, index=False)
                    print(f"  saved {path} ({len(agg_df)} rows)")
                out.update(dice_aggs)

        chart_paths = self._make_charts(df, dice_df)
        self._log_to_wandb(df, dice_df, out, chart_paths)
        return out

    # ------------------------------------------------------------------
    # Charting + W&B
    # ------------------------------------------------------------------

    def _make_charts(self, df, dice_df=None):
        """Render per-metric charts to PNG; return list of paths.

        Image-metric charts come from `df` (per_config_metrics).
        If `dice_df` is provided, also produce mean-Dice and mean-vol-diff
        charts (averaged across ROIs per (subject, config)).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        agg_key = self.experiment.aggregation_key
        plot_kind = self.experiment.plot_kind
        metric_cols = [c for c in ["ssim", "psnr", "mse", "nrmse", "epe"] if c in df.columns]

        chart_paths = []
        schemes = [
            ("by_group", ["group"]),
            ("by_group_age", ["group", "age_bin"]),
        ]
        x_values = self._x_axis_order(df[agg_key])

        def _plot(metric_label, source_df, value_col, suffix, hue_cols):
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped = (
                source_df.groupby(hue_cols + [agg_key])[value_col]
                .agg(["mean", "std"]).reset_index()
            )
            if plot_kind == "bar":
                self._draw_bars(ax, grouped, hue_cols, agg_key, value_col, x_values)
            else:
                self._draw_lines(ax, grouped, hue_cols, agg_key, value_col, x_values)
            ax.set_xlabel(agg_key)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{self.experiment.name} — {metric_label} ({suffix})")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(charts_dir, f"{value_col}_{suffix}.png")
            fig.savefig(path, dpi=110)
            plt.close(fig)
            return path

        # Image metrics: 5 metrics × 2 stratifications.
        for metric in metric_cols:
            for suffix, hue_cols in schemes:
                chart_paths.append(_plot(metric, df, metric, suffix, hue_cols))

        # Dice / volume: avg-across-ROIs per (subject, config), then group as usual.
        if dice_df is not None and not dice_df.empty:
            keys = ["subject", "group", "age_bin", agg_key]
            roi_avg = (
                dice_df.groupby(keys, dropna=False)[["dice", "vol_diff_pct"]]
                .mean().reset_index()
            )
            for value_col, label in [
                ("dice", "mean dice (avg over ROIs)"),
                ("vol_diff_pct", "mean |vol_diff_pct| (avg over ROIs)"),
            ]:
                # Use the absolute value for vol_diff_pct so signs don't cancel.
                source = roi_avg.copy()
                if value_col == "vol_diff_pct":
                    source[value_col] = source[value_col].abs()
                for suffix, hue_cols in schemes:
                    chart_paths.append(_plot(label, source, value_col, suffix, hue_cols))

            # Also: per-region Dice at the "extreme" agg_key value (worst case).
            extreme = self._extreme_x_value(x_values)
            sub = dice_df[dice_df[agg_key] == extreme]
            if not sub.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                pivot = (
                    sub.groupby(["group", "region"])["dice"].mean().unstack("group")
                )
                pivot.plot(kind="bar", ax=ax, alpha=0.85)
                ax.set_xlabel("ROI")
                ax.set_ylabel("mean Dice")
                ax.set_title(
                    f"{self.experiment.name} — Dice per ROI at "
                    f"{agg_key}={extreme}"
                )
                ax.set_xticklabels([t.get_text().replace("_", " ") for t in ax.get_xticklabels()],
                                   rotation=30, ha="right")
                ax.legend(title="group", fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
                fig.tight_layout()
                path = os.path.join(charts_dir, f"dice_per_region_at_extreme_{extreme}.png")
                fig.savefig(path, dpi=110)
                plt.close(fig)
                chart_paths.append(path)

        print(f"  saved {len(chart_paths)} charts under {charts_dir}")
        return chart_paths

    @staticmethod
    def _extreme_x_value(x_values):
        """Pick the 'most-degraded' x value: max for numeric, last otherwise."""
        try:
            return max(x_values, key=float)
        except (TypeError, ValueError):
            return x_values[-1]

    @staticmethod
    def _x_axis_order(series):
        """Return sorted unique x-values; numeric sort if all numeric, else lexicographic.

        For Exp 2, prefer a chronological order of masked timepoints over
        alphabetical ('drop_m06', 'drop_m12', ..., 'full' last).
        """
        vals = list(series.dropna().unique())
        try:
            return sorted(vals, key=float)
        except (TypeError, ValueError):
            tp_order = {"none": -1, "sc": 0, "m06": 6, "m12": 12, "m18": 18, "m24": 24, "m36": 36}
            if all(v in tp_order for v in vals):
                return sorted(vals, key=lambda v: tp_order[v])
            return sorted(vals)

    @staticmethod
    def _draw_lines(ax, grouped, hue_cols, agg_key, metric, x_values):
        """Line plot with mean + std band, one line per hue combination."""
        for hue_vals, sub in grouped.groupby(hue_cols):
            label = "/".join(map(str, hue_vals)) if isinstance(hue_vals, tuple) else str(hue_vals)
            sub = sub.set_index(agg_key).reindex(x_values).reset_index()
            ax.errorbar(
                sub[agg_key], sub["mean"], yerr=sub["std"].fillna(0),
                marker="o", capsize=3, label=label, alpha=0.85,
            )

    @staticmethod
    def _draw_bars(ax, grouped, hue_cols, agg_key, metric, x_values):
        """Grouped bar plot, one bar series per hue combination."""
        import numpy as _np
        sub_by_hue = {
            hue: sub.set_index(agg_key).reindex(x_values).reset_index()
            for hue, sub in grouped.groupby(hue_cols)
        }
        n = len(sub_by_hue)
        bar_w = 0.8 / max(n, 1)
        x = _np.arange(len(x_values))

        for i, (hue_vals, sub) in enumerate(sub_by_hue.items()):
            label = "/".join(map(str, hue_vals)) if isinstance(hue_vals, tuple) else str(hue_vals)
            ax.bar(
                x + i * bar_w, sub["mean"].fillna(0), bar_w,
                yerr=sub["std"].fillna(0), capsize=3, label=label, alpha=0.85,
            )
        ax.set_xticks(x + bar_w * (n - 1) / 2)
        ax.set_xticklabels(x_values, rotation=30, ha="right")

    def _log_to_wandb(self, per_config_df, dice_df, agg_dfs, chart_paths):
        """Log per-config + summary tables, dice tables, and chart images to W&B."""
        if self.cfg.system.debug or not self.cfg.wandb.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("  W&B not installed; skipping logging.")
            return

        proj = f"{self.cfg.wandb.project}_Robustness"
        short_run = self.cfg.run_name[-30:] if len(self.cfg.run_name) > 30 else self.cfg.run_name
        run = wandb.init(
            project=proj,
            entity=self.cfg.wandb.entity,
            name=f"{self.experiment.name}_fold{self.cfg.data.fold}_{short_run}",
            config={
                "experiment": self.experiment.name,
                "fold": self.cfg.data.fold,
                "run_name": self.cfg.run_name,
                "aggregation_key": self.experiment.aggregation_key,
                "plot_kind": self.experiment.plot_kind,
                "with_dice": dice_df is not None and not dice_df.empty,
            },
            reinit=True,
        )

        run.log({"per_config_table": wandb.Table(dataframe=per_config_df)})
        if dice_df is not None and not dice_df.empty:
            run.log({"per_config_dice_volume_table": wandb.Table(dataframe=dice_df)})
        for name, df in agg_dfs.items():
            run.log({f"{name}_table": wandb.Table(dataframe=df)})

        # Summary scalars from overall image-metric agg.
        overall = agg_dfs.get("overall_per_config")
        if overall is not None:
            for _, row in overall.iterrows():
                key = row[self.experiment.aggregation_key]
                for metric in ["ssim", "psnr", "mse", "nrmse", "epe"]:
                    col = f"{metric}_mean"
                    if col in row:
                        run.summary[f"overall_{metric}_at_{key}"] = float(row[col])

        # Summary scalars from overall Dice agg (averaged across ROIs).
        dice_overall = agg_dfs.get("dice_overall_region_config")
        if dice_overall is not None and not dice_overall.empty:
            agg_key = self.experiment.aggregation_key
            avg_across_rois = (
                dice_overall.groupby(agg_key)["dice_mean"].mean().reset_index()
            )
            for _, row in avg_across_rois.iterrows():
                run.summary[f"overall_dice_at_{row[agg_key]}"] = float(row["dice_mean"])

        for path in chart_paths:
            tag = os.path.splitext(os.path.basename(path))[0]
            run.log({f"chart/{tag}": wandb.Image(path)})

        run.finish()
        print(f"  logged tables + {len(chart_paths)} charts to W&B project '{proj}'")

    # ------------------------------------------------------------------
    # Phase 3: cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Delete .nii.gz files in the output directory; keep CSVs."""
        n = 0
        for f in os.listdir(self.output_dir):
            if f.endswith(".nii.gz"):
                os.remove(os.path.join(self.output_dir, f))
                n += 1
        print(f"  removed {n} .nii.gz files from {self.output_dir}")

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self, skip_inference=False, skip_existing=False,
            recover_metrics=False, with_dice=False, cleanup=False):
        """End-to-end driver.

        Phase 1 selection (mutually exclusive):
          - skip_inference=True   -> skip phase 1 entirely (read existing CSV)
          - recover_metrics=True  -> rebuild CSV from saved .nii.gz files (no model)
          - both False            -> run full inference (the default)

        Phase 1.5:
          - with_dice=True        -> compute Dice + volume from saved .nii.gz before agg

        Phase 2 (always): aggregation + charts + W&B.
        Phase 3 (optional): cleanup deletes .nii.gz, keeps CSVs.
        """
        if recover_metrics and skip_inference:
            raise ValueError("--recover-metrics and --skip-inference are mutually exclusive.")
        if recover_metrics:
            self.recover_metrics()
        elif not skip_inference:
            self.run_inference(skip_existing=skip_existing)
        if with_dice:
            self.run_dice_eval()
        results = self.run_aggregation()
        if cleanup:
            self.cleanup()
        return results
