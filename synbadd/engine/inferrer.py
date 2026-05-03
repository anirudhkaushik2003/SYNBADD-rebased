"""Subject-wise inference engine: exports .nii.gz synthetics and metrics CSV.

Supports three inference methods via the `method` parameter:
    - "synbadd" (default): Full VAE encode -> temporal model -> VAE decode -> generator
    - "avg_impute": Average all available SVFs -> generator (no VAE/transformer)
    - "forward_impute": Copy most recent previous scan as prediction (no generator)

Usage:
    synbadd infer --config experiment.yaml --fold 1 --method synbadd
    synbadd infer --config experiment.yaml --fold 1 --method avg_impute
    synbadd infer --config experiment.yaml --fold 1 --method forward_impute
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from synbadd.engine.checkpoint import CheckpointManager, adapt_state_dict
from synbadd.losses.similarity import compute_metrics
from synbadd.models.registry import build_model


def _compute_epe(target_svf, synthetic_field):
    """L2 end-point error between target and predicted SVF."""
    diff = target_svf - synthetic_field
    return torch.sqrt((diff ** 2).sum(dim=1)).mean().item()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_inferrer(cfg, test_loader=None, method="synbadd"):
    """Create the appropriate inferrer for the given method.

    Args:
        cfg: SynbaddConfig.
        test_loader: DataLoader for test set (not needed for forward_impute).
        method: "synbadd", "avg_impute", or "forward_impute".

    Returns:
        An inferrer instance with a .run() method.
    """
    if method == "forward_impute":
        return ForwardImputeInferrer(cfg)
    return Inferrer(cfg, test_loader, method=method)


# ---------------------------------------------------------------------------
# Inferrer (handles synbadd + avg_impute)
# ---------------------------------------------------------------------------

class Inferrer:
    """Loads checkpoints and runs subject-wise inference, saving .nii.gz outputs.

    Supports two modes:
        - method="synbadd": VAE encode -> temporal model -> VAE decode -> generator
        - method="avg_impute": averaged SVFs -> generator (no VAE/transformer needed)

    For each test subject, saves:
        - {basename}_{target_tp}_synthetic.nii.gz   (generated scan)
        - {basename}_{target_tp}_synthetic_field.nii.gz  (generated SVF)

    Also saves a per-timepoint metrics CSV and logs to W&B.
    """

    TIMEPOINTS = ["sc", "m06", "m12", "m18", "m24", "m36"]

    def __init__(self, cfg, test_loader, method="synbadd"):
        if method not in ("synbadd", "avg_impute"):
            raise ValueError(f"Inferrer supports 'synbadd' or 'avg_impute', got '{method}'")

        self.cfg = cfg
        self.method = method
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timepoint_map = {i: tp for i, tp in enumerate(self.TIMEPOINTS)}

        if method == "avg_impute":
            self._build_generator_only()
            self._load_generator_checkpoint()
        else:
            self._build_models()
            self._load_checkpoints()
        self._build_lddmm()

        # Output directory
        if method == "avg_impute":
            run_dir = f"FINAL_Avg_impute_fold_{cfg.data.fold}"
        else:
            run_dir = cfg.run_name
        self.output_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{cfg.data.fold}",
            run_dir,
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_generator_only(self):
        """Build only the generator (for avg_impute mode)."""
        cfg = self.cfg
        image_size = (cfg.model.img_height,) * 3

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
            using_unet=False,  # avg impute doesn't use UNet
        )

        self.vae_model = None
        self.temporal_model = None
        self.discriminator = None

        if cfg.system.n_gpu > 1:
            self.generator = nn.DataParallel(self.generator)
        self.generator = self.generator.to(self.device)

    def _build_models(self):
        """Build all models (for synbadd mode)."""
        cfg = self.cfg
        image_size = (cfg.model.img_height,) * 3

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

        self.vae_model = build_model("VAE3D_UNetSkip")
        sd = torch.load(cfg.paths.vae_checkpoint, map_location="cpu", weights_only=False)
        sd = adapt_state_dict(sd, self.vae_model)
        self.vae_model.load_state_dict(sd)
        for p in self.vae_model.parameters():
            p.requires_grad = False

        self.temporal_model = build_model(
            cfg.model.temporal.type,
            input_dim=cfg.model.latent_dim,
            num_heads=cfg.model.temporal.num_heads,
            num_layers=cfg.model.temporal.num_layers,
            max_seq_len=cfg.model.temporal.max_seq_len,
            use_positional_embeddings=cfg.model.temporal.use_positional_embeddings,
            use_bidirectional_embeddings=cfg.model.temporal.use_bidirectional_embeddings,
        )

        self.discriminator = None
        if cfg.model.discriminator.enabled:
            self.discriminator = build_model(
                cfg.model.discriminator.type,
                in_channels=1,
                ndim_non_img=cfg.model.c_dim,
                channel_number=cfg.model.discriminator.channel_number,
            )

        # DataParallel
        if cfg.system.n_gpu > 1:
            self.generator = nn.DataParallel(self.generator)
            self.vae_model = nn.DataParallel(self.vae_model)
            self.temporal_model = nn.DataParallel(self.temporal_model)
            if self.discriminator is not None:
                self.discriminator = nn.DataParallel(self.discriminator)

        self.generator = self.generator.to(self.device)
        self.vae_model = self.vae_model.to(self.device)
        self.temporal_model = self.temporal_model.to(self.device)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device)

    def _load_generator_checkpoint(self):
        """Load only the generator checkpoint (for avg_impute)."""
        ckpt = CheckpointManager(
            self.cfg.paths.checkpoint_dir, self.cfg.project_name, self.cfg.run_name
        )
        ckpt.load({"generator": self.generator}, device=self.device)

    def _load_checkpoints(self):
        """Load all model checkpoints (for synbadd)."""
        ckpt = CheckpointManager(
            self.cfg.paths.checkpoint_dir, self.cfg.project_name, self.cfg.run_name
        )
        models = {"generator": self.generator, "temporal": self.temporal_model}
        if self.discriminator is not None:
            models["discriminator"] = self.discriminator
        ckpt.load(models, device=self.device)

    def _build_lddmm(self):
        cfg = self.cfg
        image_size = (cfg.model.img_height,) * 3
        from voxelmorph.torch.networks import VxmDense

        from synbadd.losses.lddmm import VxmLDDMMLoss

        vxm_model = VxmDense(
            inshape=image_size,
            nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32]],
        )
        vxm_model.to(self.device)
        sd = torch.load(cfg.paths.vxm_checkpoint, map_location=self.device, weights_only=False)
        sd = adapt_state_dict(sd, vxm_model)
        vxm_model.load_state_dict(sd)
        vxm_model.eval()
        for p in vxm_model.parameters():
            p.requires_grad = False
        if cfg.system.n_gpu > 1:
            vxm_model = nn.DataParallel(vxm_model)
        self.lddmm_loss = VxmLDDMMLoss(vxm_model)

    def _get_vae_module(self):
        """Get underlying VAE module (unwrap DataParallel if needed)."""
        if hasattr(self.vae_model, "module"):
            return self.vae_model.module
        return self.vae_model

    # ------------------------------------------------------------------
    # Contextualization strategies
    # ------------------------------------------------------------------

    def _contextualize_synbadd(self, stacked_svf, target_timepoint_index, latest_scan):
        """Full VAE encode -> temporal model -> VAE decode path."""
        cfg = self.cfg
        if cfg.data.use_past_data or cfg.data.use_future_data:
            batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
            stacked_flat = stacked_svf.view(-1, c_dim, h, w, d)

            vae = self._get_vae_module()
            mu, logvar, skips = vae.encode(stacked_flat)
            mu, logvar = mu.detach(), logvar.detach()
            skips = [s.detach() for s in skips]

            # Average skips across temporal dimension
            for si in range(len(skips)):
                s_shape = skips[si].shape
                skips[si] = skips[si].view(batch_size, temporal_dim, *s_shape[1:]).mean(dim=1)

            latent = mu.view(batch_size, temporal_dim, -1)
            contextualized = self.temporal_model(latent, target_timepoint_index)
            return vae.decode(contextualized, skips)
        else:
            return latest_scan

    def _contextualize_avg_impute(self, stacked_svf, target_timepoint_index, latest_scan):
        """Average SVFs across temporal dimension."""
        return stacked_svf.mean(dim=1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Run subject-wise inference, saving .nii.gz and metrics CSV."""
        cfg = self.cfg
        method_label = self.method
        wandb = self._init_wandb(method_label)
        uid_to_filepath = self.test_loader.dataset.get_file_path_mapping()

        self.generator.eval()
        if self.vae_model is not None:
            self.vae_model.eval()
        if self.temporal_model is not None:
            self.temporal_model.eval()
        if self.discriminator is not None:
            self.discriminator.eval()

        # Select contextualization strategy
        if self.method == "avg_impute":
            contextualize_fn = self._contextualize_avg_impute
        else:
            contextualize_fn = self._contextualize_synbadd

        all_metrics = []
        tp_metrics = {tp: [] for tp in self.TIMEPOINTS}
        subject_idx = 0

        with torch.no_grad():
            with tqdm(total=len(self.test_loader.dataset), desc=f"Inference ({method_label})") as pbar:
                for batch in self.test_loader:
                    device = self.device
                    stacked_svf = batch["stacked_svf"].to(device).float()
                    latest_scan = batch["latest_scan"].to(device).float()
                    target_scan = batch["target_scan"].to(device).float()
                    class_labels = batch["class_label"].to(device).float()
                    reg_labels = batch["reg_label"].to(device).float()
                    target_timepoint = batch["target_timepoint"].to(device).float()
                    target_timepoint_index = batch["target_timepoint_index"].to(device)
                    target_svf = batch["target_svf"].to(device).float()
                    target_uid = batch["target_uid"].to(device)

                    target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)

                    # Compute contextualized scan using the selected strategy
                    contextualized_scan = contextualize_fn(
                        stacked_svf, target_timepoint_index, latest_scan
                    )

                    y_synthetic, synthetic_field = self.generator(
                        latest_scan, contextualized_scan, target_labels
                    )

                    # Compute metrics
                    epe = _compute_epe(target_svf, synthetic_field)
                    mse, psnr_val, ssim_val, mean_jac, std_jac, _, nrmse, mssim = compute_metrics(
                        target_scan, y_synthetic, synthetic_field, self.lddmm_loss, device
                    )

                    # Per-subject metrics
                    for b in range(target_uid.size(0)):
                        uid = target_uid[b].item()
                        filepath = uid_to_filepath.get(uid)
                        basename = os.path.basename(filepath) if filepath else f"subject_{uid}"
                        tp_idx = target_timepoint_index[b].item()
                        tp = self.timepoint_map.get(tp_idx + 1, "unknown")

                        row = {
                            "subject": basename,
                            "timepoint": tp,
                            "mse": mse, "psnr": psnr_val, "ssim": ssim_val,
                            "mssim": mssim, "mean_jac": mean_jac, "std_jac": std_jac,
                            "nrmse": nrmse, "epe": epe,
                        }
                        all_metrics.append(row)

                        if tp in tp_metrics:
                            tp_metrics[tp].append(row)

                        # Log per-subject to W&B
                        if wandb is not None:
                            wandb.log(
                                {f"Infer {k}": v for k, v in row.items() if k not in ("subject", "timepoint")},
                                step=subject_idx,
                            )
                            subject_idx += 1

                        # Save .nii.gz
                        if filepath is None:
                            continue

                        tp_label = self.timepoint_map.get(
                            tp_idx + 1, cfg.data.target_timepoint,
                        )
                        syn_name = basename.replace(".nii.gz", f"_{tp_label}_synthetic.nii.gz")
                        field_name = basename.replace(".nii.gz", f"_{tp_label}_synthetic_field.nii.gz")

                        nib.save(
                            nib.Nifti1Image(y_synthetic[b].cpu().numpy(), None),
                            os.path.join(self.output_dir, syn_name),
                        )
                        nib.save(
                            nib.Nifti1Image(synthetic_field[b].cpu().numpy(), None),
                            os.path.join(self.output_dir, field_name),
                        )

                    pbar.update(stacked_svf.size(0))
                    pbar.set_postfix({"ssim": ssim_val, "epe": epe})

        # Save CSV
        self._save_csv(all_metrics, tp_metrics)

        # W&B summary
        if wandb is not None and all_metrics:
            avg = {k: np.mean([r[k] for r in all_metrics])
                   for k in all_metrics[0] if k not in ("subject", "timepoint")}
            wandb.summary.update({f"Infer avg_{k}": v for k, v in avg.items()})

            # Per-timepoint summary
            for tp, rows in tp_metrics.items():
                if rows:
                    tp_avg = {k: np.mean([r[k] for r in rows])
                              for k in rows[0] if k not in ("subject", "timepoint")}
                    wandb.summary.update({f"Infer {tp}_avg_{k}": v for k, v in tp_avg.items()})

            # Log full results as W&B Table
            table = wandb.Table(
                columns=["subject", "timepoint", "mse", "psnr", "ssim", "mssim", "nrmse", "epe"],
                data=[[r["subject"], r["timepoint"], r["mse"], r["psnr"], r["ssim"],
                       r["mssim"], r["nrmse"], r["epe"]] for r in all_metrics],
            )
            wandb.log({"Infer results": table})
            wandb.finish()

        n_subjects = len(all_metrics)
        avg_loss = np.mean([r["mse"] for r in all_metrics]) if all_metrics else 0.0
        print(f"Inference ({method_label}) complete. {n_subjects} subjects. Avg MSE: {avg_loss:.4f}")
        print(f"Outputs saved to {self.output_dir}")
        return {"n_subjects": n_subjects, "output_dir": self.output_dir, "avg_mse": avg_loss}

    def _save_csv(self, all_metrics, tp_metrics):
        """Save overall and per-timepoint metrics CSV."""
        if not all_metrics:
            return

        save_dir = os.path.join(
            self.cfg.paths.checkpoint_dir, self.cfg.project_name, "infer"
        )
        os.makedirs(save_dir, exist_ok=True)

        numeric_keys = [k for k in all_metrics[0] if k not in ("subject", "timepoint")]
        avg = {k: np.mean([r[k] for r in all_metrics]) for k in numeric_keys}
        for tp, rows in tp_metrics.items():
            if rows:
                for k in numeric_keys:
                    avg[f"{tp}_{k}"] = np.mean([r[k] for r in rows])

        label = "avg_impute" if self.method == "avg_impute" else self.cfg.run_name

        # Per-subject CSV
        per_subject_df = pd.DataFrame(all_metrics)
        per_subject_path = os.path.join(save_dir, f"{label}_per_subject.csv")
        per_subject_df.to_csv(per_subject_path, index=False)
        print(f"Per-subject metrics saved to {per_subject_path}")

        # Summary CSV
        df = pd.DataFrame([avg])
        csv_path = os.path.join(save_dir, f"{label}_infer.csv")
        df.to_csv(csv_path, index=False)
        print(f"Summary metrics saved to {csv_path}")

    def _init_wandb(self, method_label):
        if self.cfg.system.debug or not self.cfg.wandb.enabled:
            return None
        import wandb

        run_name = self.cfg.run_name
        if method_label != "synbadd":
            run_name = f"FINAL_{method_label}_fold_{self.cfg.data.fold}"
        wandb.init(
            project=f"{self.cfg.wandb.project}_Subjectwise",
            entity=self.cfg.wandb.entity,
            name=run_name,
        )
        return wandb


# ---------------------------------------------------------------------------
# Forward impute inferrer (standalone — no generator, no checkpoints)
# ---------------------------------------------------------------------------

class ForwardImputeInferrer:
    """Forward imputation baseline: use the most recent previous scan as prediction.

    For each test subject and target timepoint, finds the most recent available
    scan before the target timepoint and copies it as the prediction. If m24 is
    the target and m18 is missing, it uses m12 (or m06, etc.).

    No generator or checkpoints needed — this is a pure copy baseline.
    """

    VISIT_ORDER = ["sc", "m06", "m12", "m18", "m24", "m36"]

    def __init__(self, cfg):
        self.cfg = cfg
        self.fold = cfg.data.fold

        # Load and filter CSV to test fold
        self.df = pd.read_csv(cfg.paths.csv_file)
        self.df["Acq Date"] = pd.to_datetime(self.df["Acq Date"])
        self.df = (
            self.df.sort_values(by=["Subject", "Visit", "Acq Date"])
            .drop_duplicates(subset=["Subject", "Visit"], keep="last")
        )
        self.df = self.df[self.df[f"fold_{self.fold}"] == "test"].reset_index(drop=True)

        # Output directory
        self.output_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{self.fold}",
            f"FINAL_Forward_impute_fold_{self.fold}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_most_recent_scan(self, subject_id, target_visit):
        """Find the most recent available scan before target_visit."""
        target_idx = self.VISIT_ORDER.index(target_visit)
        subject_data = self.df[self.df["Subject"] == subject_id]

        for idx in range(target_idx - 1, -1, -1):
            visit = self.VISIT_ORDER[idx]
            match = subject_data[subject_data["Visit"] == visit]
            if not match.empty:
                return match.iloc[0]
        return None

    def _load_nii(self, path):
        """Load a NIfTI file, pad to 128^3, normalize to [0,1]."""
        img = nib.load(path).get_fdata()
        # Pad to 128^3 if needed
        target_shape = (128, 128, 128)
        if img.shape != target_shape:
            padded = np.zeros(target_shape)
            slices = tuple(slice(0, min(s, t)) for s, t in zip(img.shape, target_shape))
            padded[slices] = img[slices]
            img = padded
        # Min-max normalize
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img

    def run(self):
        """Run forward imputation: copy most recent scan as prediction."""
        cfg = self.cfg
        wandb = self._init_wandb()

        subjects = self.df.groupby("Subject")
        metrics_rows = []
        n_saved = 0

        target_tp_mode = cfg.data.target_timepoint
        if target_tp_mode == "all":
            target_visits = self.VISIT_ORDER[1:]  # all except sc
        else:
            target_visits = [target_tp_mode]

        with tqdm(total=len(subjects), desc="Forward impute") as pbar:
            for subject_id, group in subjects:
                for target_visit in target_visits:
                    # Skip if subject doesn't have this target visit
                    if target_visit not in group["Visit"].values:
                        continue
                    if target_visit == "sc":
                        continue

                    prev_row = self._get_most_recent_scan(subject_id, target_visit)
                    if prev_row is None:
                        continue

                    target_row = group[group["Visit"] == target_visit].iloc[0]
                    target_path = target_row["File_Path"]
                    forward_path = prev_row["File_Path"]

                    if not os.path.exists(forward_path) or not os.path.exists(target_path):
                        continue

                    # Load scans
                    target_img = self._load_nii(target_path)
                    forward_img = self._load_nii(forward_path)

                    # Compute SSIM/PSNR
                    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

                    ssim_val = structural_similarity(
                        target_img, forward_img, data_range=1.0
                    )
                    psnr_val = peak_signal_noise_ratio(
                        target_img, forward_img, data_range=1.0
                    )
                    mse_val = float(np.mean((target_img - forward_img) ** 2))

                    metrics_rows.append({
                        "subject": subject_id,
                        "target_visit": target_visit,
                        "prev_visit": prev_row["Visit"],
                        "ssim": ssim_val,
                        "psnr": psnr_val,
                        "mse": mse_val,
                    })

                    # Save forward-imputed scan as synthetic .nii.gz
                    basename = os.path.basename(target_path)
                    syn_name = basename.replace(
                        ".nii.gz", f"_{target_visit}_synthetic.nii.gz"
                    )
                    nib.save(
                        nib.Nifti1Image(forward_img.astype(np.float32), None),
                        os.path.join(self.output_dir, syn_name),
                    )
                    n_saved += 1

                pbar.update(1)

        # Save metrics CSV
        self._save_csv(metrics_rows)

        # W&B
        if wandb is not None and metrics_rows:
            avg = {
                "ssim": np.mean([r["ssim"] for r in metrics_rows]),
                "psnr": np.mean([r["psnr"] for r in metrics_rows]),
                "mse": np.mean([r["mse"] for r in metrics_rows]),
            }
            wandb.log({f"Forward Impute {k}": v for k, v in avg.items()})
            wandb.finish()

        print(f"Forward impute complete. {n_saved} scans saved to {self.output_dir}")
        return {"n_subjects": n_saved, "output_dir": self.output_dir}

    def _save_csv(self, metrics_rows):
        """Save forward impute metrics CSV."""
        if not metrics_rows:
            return
        save_dir = os.path.join(
            self.cfg.paths.checkpoint_dir, "forward_impute"
        )
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(metrics_rows)
        csv_path = os.path.join(save_dir, f"forward_impute_fold_{self.fold}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Forward impute metrics saved to {csv_path}")

    def _init_wandb(self):
        if self.cfg.system.debug or not self.cfg.wandb.enabled:
            return None
        import wandb

        wandb.init(
            project=f"{self.cfg.wandb.project}_Subjectwise",
            entity=self.cfg.wandb.entity,
            name=f"FINAL_Forward_impute_fold_{self.fold}",
        )
        return wandb
