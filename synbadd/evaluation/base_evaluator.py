"""Base evaluator ABC for all comparison/evaluation types."""

import os
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Abstract base class for evaluation comparisons.

    Subclasses implement compute() which returns a dict of DataFrames.
    Base class provides report() (save CSVs) and log_wandb().
    """

    def __init__(self, cfg, fold):
        self.cfg = cfg
        self.fold = fold
        self.roi_regions = cfg.eval.roi_regions

        self.output_dir = os.path.join(
            cfg.paths.output_dir,
            f"SYNBADD_Qualitative_{fold}",
            "evaluation",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def compute(self):
        """Run evaluation and return dict of {name: DataFrame}.

        Returns:
            dict mapping CSV name (without .csv) -> pd.DataFrame.
        """

    def report(self, results=None):
        """Save all result DataFrames as CSVs."""
        if results is None:
            results = self.compute()

        for name, df in results.items():
            path = os.path.join(self.output_dir, f"{name}_fold_{self.fold}.csv")
            df.to_csv(path, index=False)
            print(f"Saved: {path}")

        return results

    def log_wandb(self, results=None):
        """Log results to W&B as tables."""
        if self.cfg.system.debug or not self.cfg.wandb.enabled:
            return

        try:
            import wandb
        except ImportError:
            return

        if results is None:
            results = self.compute()

        for name, df in results.items():
            table = wandb.Table(dataframe=df)
            wandb.log({f"{name}_fold_{self.fold}": table})
