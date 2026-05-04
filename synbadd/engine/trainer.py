"""Training engine for the SYNBADD pipeline."""



import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from voxelmorph.torch.networks import VxmDense

from synbadd.engine.checkpoint import CheckpointManager, adapt_state_dict
from synbadd.engine.validator import Validator
from synbadd.losses.composite import build_loss_fn
from synbadd.losses.lddmm import VxmLDDMMLoss
from synbadd.models.registry import build_model


class Trainer:
    """Orchestrates training for the full SYNBADD pipeline.

    Handles model instantiation, optimizer setup, training loop with
    gradient accumulation, validation, early stopping, checkpointing,
    and W&B logging.
    """

    def __init__(self, cfg, train_loader, val_loader=None, test_loader=None):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build all models
        self._build_models()
        # Build optimizers + schedulers
        self._build_optimizers()
        # Build loss function
        self._build_loss()
        # Checkpoint manager
        self.ckpt = CheckpointManager(
            cfg.paths.checkpoint_dir, cfg.project_name, cfg.run_name
        )
        # Validator
        self.validator = Validator(cfg, self.device)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_models(self):
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

        # VAE (frozen encoder)
        self.vae_model = self._load_frozen_vae()

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

        # Discriminator (optional)
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

        # Move to device
        self.generator = self.generator.to(self.device)
        self.vae_model = self.vae_model.to(self.device)
        self.temporal_model = self.temporal_model.to(self.device)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device)

        # Keep VAE frozen end-to-end: avoids BN running stats drifting during training
        self.vae_model.eval()

    def _get_vae_module(self):
        """Return the underlying VAE (unwrap DataParallel if needed)."""
        return self.vae_model.module if hasattr(self.vae_model, "module") else self.vae_model

    def _load_frozen_vae(self):
        """Load pretrained VAE and freeze all parameters."""
        vae = build_model("VAE3D_UNetSkip")
        sd = torch.load(self.cfg.paths.vae_checkpoint, map_location="cpu", weights_only=False)
        sd = adapt_state_dict(sd, vae)
        vae.load_state_dict(sd)
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        return vae

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizers(self):
        cfg = self.cfg
        lr = cfg.training.lr
        betas = tuple(cfg.training.betas)
        patience = cfg.training.patience
        min_lr = cfg.training.min_lr

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, mode="min", factor=0.5, patience=patience, min_lr=min_lr
        )

        self.optimizer_T = torch.optim.Adam(self.temporal_model.parameters(), lr=lr, betas=betas)
        self.scheduler_T = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_T, mode="min", factor=0.5, patience=patience, min_lr=min_lr
        )

        self.optimizer_D = None
        self.scheduler_D = None
        if self.discriminator is not None:
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
            self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D, mode="min", factor=0.5, patience=patience, min_lr=min_lr
            )

    # ------------------------------------------------------------------
    # Loss construction
    # ------------------------------------------------------------------

    def _build_loss(self):
        """Build LDDMM loss (pretrained VxmDense) and composite loss."""
        cfg = self.cfg
        image_size = (cfg.model.img_height,) * 3

        # Load pretrained VoxelMorph for LDDMM loss
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

        lddmm_loss = VxmLDDMMLoss(vxm_model)
        self.criterion = build_loss_fn(cfg, lddmm_loss=lddmm_loss)

    # ------------------------------------------------------------------
    # Forward pass (shared between train/val/test)
    # ------------------------------------------------------------------

    def _forward(self, batch):
        """Run forward pass on a batch, returning generated image and field."""
        device = self.device
        cfg = self.cfg
        stacked_svf = batch["stacked_svf"].to(device).float()
        latest_scan = batch["latest_scan"].to(device).float()
        class_labels = batch["class_label"].to(device).float()
        reg_labels = batch["reg_label"].to(device).float()
        target_timepoint = batch["target_timepoint"].to(device).float()
        target_timepoint_index = batch["target_timepoint_index"].to(device)

        target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)

        # VAE encode -> temporal model -> VAE decode
        if cfg.data.use_past_data or cfg.data.use_future_data:
            batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
            stacked_flat = stacked_svf.view(-1, c_dim, h, w, d)

            vae = self._get_vae_module()
            mu, _, skips = vae.encode(stacked_flat)
            mu = mu.detach()
            skips = [s.detach() for s in skips]

            for si in range(len(skips)):
                s_shape = skips[si].shape
                skips[si] = skips[si].view(batch_size, temporal_dim, *s_shape[1:]).mean(dim=1)

            latent = mu.view(batch_size, temporal_dim, -1)
            contextualized = self.temporal_model(latent, target_timepoint_index)
            contextualized_scan = vae.decode(contextualized, skips)
        else:
            contextualized_scan = latest_scan

        y_synthetic, synthetic_field = self.generator(
            latest_scan, contextualized_scan, target_labels
        )
        return y_synthetic, synthetic_field, target_labels

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Full training loop with early stopping and checkpointing."""
        cfg = self.cfg
        wandb = self._init_wandb()

        accumulation_steps = max(1, cfg.training.effective_batch_size // cfg.training.batch_size)
        best_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = cfg.training.early_stopping_patience

        for epoch in range(cfg.training.n_epochs):
            epoch_loss, train_metrics = self._train_epoch(accumulation_steps)

            # Validation
            if self.val_loader is not None and cfg.data.use_val_dataset:
                val_loss = self.validator.run(
                    self.generator, self.vae_model, self.temporal_model,
                    self.discriminator, self.val_loader, self.criterion
                )
            else:
                val_loss = epoch_loss

            # LR scheduling
            self.scheduler_G.step(val_loss)
            self.scheduler_T.step(val_loss)
            if self.scheduler_D is not None:
                self.scheduler_D.step(val_loss)

            # W&B logging
            if wandb is not None:
                log_data = {"Epoch": epoch, "Train Loss": epoch_loss, "Val Loss": val_loss,
                            "LR": self.optimizer_G.param_groups[0]["lr"]}
                log_data.update({f"Train {k}": v for k, v in train_metrics.items()})
                wandb.log(log_data)

            print(f"Epoch {epoch+1}/{cfg.training.n_epochs}: Train={epoch_loss:.4f}, Val={val_loss:.4f}")

            # Checkpointing
            if best_loss - val_loss > 1e-4:
                best_loss = val_loss
                patience_counter = 0
                self.ckpt.save(self._models_dict())
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if self.optimizer_G.param_groups[0]["lr"] < cfg.training.min_lr:
                print("LR too low. Stopping.")
                break
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best and test
        if wandb is not None:
            wandb.log({"Best Validation Loss": best_loss})

        if self.test_loader is not None:
            self.ckpt.load(self._models_dict(), device=self.device)
            test_loss = self.validator.run(
                self.generator, self.vae_model, self.temporal_model,
                self.discriminator, self.test_loader, self.criterion,
                phase="test",
            )
            print(f"Test loss: {test_loss:.4f}")
            if wandb is not None:
                wandb.log({"Test Loss": test_loss})

        if wandb is not None:
            wandb.finish()

        return best_loss

    def _train_epoch(self, accumulation_steps):
        """Run one training epoch."""
        self.generator.train()
        self.temporal_model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        self.optimizer_G.zero_grad()
        self.optimizer_T.zero_grad()
        if self.optimizer_D is not None:
            self.optimizer_D.zero_grad()

        epoch_loss = 0.0
        metric_accum = {}
        n_batches = 0

        with tqdm(total=len(self.train_loader.dataset),
                  desc="Training") as pbar:
            for i, batch in enumerate(self.train_loader):
                target_scan = batch["target_scan"].to(self.device).float()
                target_svf = batch["target_svf"].to(self.device).float()

                y_synthetic, synthetic_field, target_labels = self._forward(batch)

                # Discriminator
                disc_out, x_real = None, None
                if self.discriminator is not None:
                    latest_scan = batch["latest_scan"].to(self.device).float()
                    disc_out = self.discriminator(latest_scan, target_labels)
                    x_real = latest_scan

                total_loss, loss_dict = self.criterion(
                    y_synthetic, target_scan, synthetic_field,
                    target_svf=target_svf, disc_out=disc_out, x_real=x_real,
                )

                total_loss.backward()

                # Gradient accumulation step
                if (i + 1) % accumulation_steps == 0:
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()
                    self.optimizer_T.step()
                    self.optimizer_T.zero_grad()
                    if self.optimizer_D is not None:
                        self.optimizer_D.step()
                        self.optimizer_D.zero_grad()

                epoch_loss += total_loss.item()
                n_batches += 1

                # Accumulate metrics
                for k, v in loss_dict.items():
                    metric_accum.setdefault(k, []).append(v)

                pbar.update(batch["stacked_svf"].size(0))
                pbar.set_postfix({"loss": total_loss.item()})

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: np.mean(v) for k, v in metric_accum.items()}
        return avg_loss, avg_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _models_dict(self):
        """Return component name -> model mapping for checkpoint manager."""
        d = {
            "generator": self.generator,
            "vae": self.vae_model,
            "temporal": self.temporal_model,
        }
        if self.discriminator is not None:
            d["discriminator"] = self.discriminator
        return d

    def _init_wandb(self):
        """Initialize W&B if enabled."""
        if self.cfg.system.debug or not self.cfg.wandb.enabled:
            return None
        import wandb
        wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.run_name,
            config=self._cfg_to_dict(),
        )
        return wandb

    def _cfg_to_dict(self):
        """Flatten config to dict for W&B."""
        from dataclasses import asdict
        return asdict(self.cfg)
