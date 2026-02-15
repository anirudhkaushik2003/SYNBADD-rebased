"""Pipeline stages: Train, Infer, Evaluate with prerequisite checking."""

import os
from abc import ABC, abstractmethod


class BaseStage(ABC):
    """Base pipeline stage with prerequisite and output checking."""

    def __init__(self, cfg, fold):
        self.cfg = cfg
        self.fold = fold

    @abstractmethod
    def check_prerequisites(self):
        """Check if all required inputs exist. Raises if not."""

    @abstractmethod
    def outputs_exist(self):
        """Check if this stage's outputs already exist (for skip-existing)."""

    @abstractmethod
    def run(self):
        """Execute the stage."""


class TrainStage(BaseStage):
    """Training stage: trains model and saves checkpoints."""

    def check_prerequisites(self):
        cfg = self.cfg
        # Need CSV file and VAE checkpoint
        if not os.path.exists(cfg.paths.csv_file):
            raise FileNotFoundError(f"CSV file not found: {cfg.paths.csv_file}")
        if not os.path.exists(cfg.paths.vae_checkpoint):
            raise FileNotFoundError(f"VAE checkpoint not found: {cfg.paths.vae_checkpoint}")
        if not os.path.exists(cfg.paths.vxm_checkpoint):
            raise FileNotFoundError(f"VxM checkpoint not found: {cfg.paths.vxm_checkpoint}")

    def outputs_exist(self):
        from synbadd.engine.checkpoint import CheckpointManager
        ckpt = CheckpointManager(
            self.cfg.paths.checkpoint_dir, self.cfg.project_name, self.cfg.run_name
        )
        return ckpt.exists("generator")

    def run(self):
        from synbadd.data.loader import build_dataloaders
        from synbadd.engine.trainer import Trainer

        loaders = build_dataloaders(self.cfg, self.fold)
        trainer = Trainer(
            self.cfg,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
        )
        return trainer.train()


class InferStage(BaseStage):
    """Inference stage: loads checkpoints and exports .nii.gz synthetics.

    Supports method="synbadd" (default), "avg_impute", or "forward_impute".
    """

    def __init__(self, cfg, fold, method="synbadd"):
        super().__init__(cfg, fold)
        self.method = method

    def check_prerequisites(self):
        # forward_impute needs no checkpoints, just the CSV
        if self.method == "forward_impute":
            if not os.path.exists(self.cfg.paths.csv_file):
                raise FileNotFoundError(f"CSV file not found: {self.cfg.paths.csv_file}")
            return

        # synbadd and avg_impute both need the generator checkpoint
        from synbadd.engine.checkpoint import CheckpointManager

        ckpt = CheckpointManager(
            self.cfg.paths.checkpoint_dir, self.cfg.project_name, self.cfg.run_name
        )
        if not ckpt.exists("generator"):
            raise FileNotFoundError(
                "Generator checkpoint not found. Run training first."
            )

    def _get_output_dir_name(self):
        if self.method == "avg_impute":
            return f"FINAL_Avg_impute_fold_{self.fold}"
        elif self.method == "forward_impute":
            return f"FINAL_Forward_impute_fold_{self.fold}"
        return self.cfg.run_name

    def outputs_exist(self):
        output_dir = os.path.join(
            self.cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{self.fold}",
            self._get_output_dir_name(),
        )
        if not os.path.isdir(output_dir):
            return False
        return any(f.endswith("_synthetic.nii.gz") for f in os.listdir(output_dir))

    def run(self):
        from synbadd.engine.inferrer import build_inferrer

        test_loader = None
        if self.method != "forward_impute":
            from synbadd.data.loader import build_dataloaders
            loaders = build_dataloaders(self.cfg, self.fold)
            test_loader = loaders["test"]

        inferrer = build_inferrer(self.cfg, test_loader=test_loader, method=self.method)
        return inferrer.run()


class EvaluateStage(BaseStage):
    """Evaluation stage: runs comparison analyses and saves CSVs."""

    def __init__(self, cfg, fold, comparison_type="noder"):
        super().__init__(cfg, fold)
        self.comparison_type = comparison_type

    def check_prerequisites(self):
        # Need inference outputs
        output_dir = os.path.join(
            self.cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{self.fold}",
            self.cfg.run_name,
        )
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(
                f"Inference outputs not found: {output_dir}. Run inference first."
            )

    def outputs_exist(self):
        eval_dir = os.path.join(
            self.cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{self.fold}",
            "evaluation",
        )
        if not os.path.isdir(eval_dir):
            return False
        return any(
            f.endswith(f"_fold_{self.fold}.csv")
            for f in os.listdir(eval_dir)
        )

    def run(self):
        from synbadd.evaluation import build_evaluator

        evaluator = build_evaluator(self.comparison_type, self.cfg)
        results = evaluator.report()
        evaluator.log_wandb(results)
        return results
