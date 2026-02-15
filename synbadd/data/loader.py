"""DataLoader utilities and factory function."""

from monai.data.dataloader import DataLoader as MonaiDataLoader

from synbadd.data.baseline_dataset import BaselineSVFDataset
from synbadd.data.piecewise_dataset import PiecewiseSVFDataset


class _RepeatSampler:
    """Sampler that repeats forever (for MultiEpochsDataLoader)."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(MonaiDataLoader):
    """DataLoader that reuses workers across epochs.

    Avoids the overhead of respawning worker processes at the start of
    every epoch, which is significant for 3D medical imaging data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


# Dataset type mapping
_DATASET_CLASSES = {
    "baseline": BaselineSVFDataset,
    "piecewise": PiecewiseSVFDataset,
}


def build_dataloaders(cfg, fold):
    """Build train/val/test dataloaders from config.

    Args:
        cfg: SynbaddConfig instance.
        fold: Fold number (1-5).

    Returns:
        dict with keys 'train', 'val', 'test' mapping to DataLoader instances.
        'val' may be None if use_val_dataset is False.
    """
    svf_type = cfg.data.svf_type
    dataset_cls = _DATASET_CLASSES.get(svf_type)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown svf_type '{svf_type}'. Choose from: {list(_DATASET_CLASSES)}"
        )

    csv_file = cfg.paths.csv_file
    batch_size = cfg.training.batch_size
    num_workers = cfg.system.n_cpu

    # Build datasets
    train_dataset = dataset_cls(cfg, csv_file, fold, subset="train")
    test_dataset = dataset_cls(cfg, csv_file, fold, subset="test")

    val_dataset = None
    if cfg.data.use_val_dataset:
        val_dataset = dataset_cls(cfg, csv_file, fold, subset="val")

    # Build loaders
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = MultiEpochsDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = MultiEpochsDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
