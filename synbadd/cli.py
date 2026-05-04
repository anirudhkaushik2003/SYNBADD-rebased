"""SYNBADD CLI — single entry point for all pipeline operations.

Usage:
    synbadd train      --config <yaml> [--fold N] [--set key=val ...]
    synbadd infer      --config <yaml> [--fold N] [--run-name NAME]
    synbadd evaluate   --config <yaml> [--fold N] [--comparison TYPE]
    synbadd pipeline   --config <yaml> [--folds 1,2,3,4,5] [--stages ...]
    synbadd manifest   query [--fold N] [--metric M] [--top N]
    synbadd manifest   show <run-name>
"""

import click

from synbadd.config.loader import load_config

# ---------------------------------------------------------------------------
# Shared option decorators
# ---------------------------------------------------------------------------

def config_options(f):
    """Common options shared across commands."""
    f = click.option("--config", "-c", "config_path", required=True, help="Path to YAML config file")(f)
    f = click.option("--fold", type=int, default=None, help="Override data.fold")(f)
    f = click.option("--set", "overrides", multiple=True, help="Override config values: key=val")(f)
    return f


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="synbadd")
def main():
    """SYNBADD: Longitudinal 3D MRI synthesis pipeline."""
    pass


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@main.command()
@config_options
def train(config_path, fold, overrides):
    """Train a model for a single fold."""
    cfg = load_config(config_path, list(overrides), fold)

    from synbadd.data.loader import build_dataloaders
    from synbadd.engine.trainer import Trainer
    from synbadd.utils import get_run_name, seed_everything

    if not cfg.run_name:
        import dataclasses
        cfg = dataclasses.replace(cfg, run_name=get_run_name(cfg))

    seed_everything(cfg.system.seed)
    loaders = build_dataloaders(cfg, cfg.data.fold)
    trainer = Trainer(
        cfg,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
    )
    result = trainer.train()
    click.echo(f"Training complete. Best val loss: {result['best_val_loss']:.6f} at epoch {result['best_epoch']}")


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

@main.command()
@config_options
@click.option("--run-name", default=None, help="Explicit run name (overrides auto-detect)")
@click.option(
    "--method", "-m", default="synbadd",
    type=click.Choice(["synbadd", "avg_impute", "forward_impute"]),
    help="Inference method: synbadd (full model), avg_impute, or forward_impute",
)
def infer(config_path, fold, overrides, run_name, method):
    """Run subject-wise inference using saved checkpoints."""
    cfg = load_config(config_path, list(overrides), fold)
    if run_name:
        cfg = _with_run_name(cfg, run_name)

    from synbadd.engine.inferrer import build_inferrer

    test_loader = None
    if method != "forward_impute":
        from synbadd.data.loader import build_dataloaders
        loaders = build_dataloaders(cfg, cfg.data.fold)
        test_loader = loaders["test"]

    inferrer = build_inferrer(cfg, test_loader=test_loader, method=method)
    result = inferrer.run()
    click.echo(f"Inference ({method}) complete. {result['n_subjects']} subjects processed.")
    click.echo(f"Outputs: {result['output_dir']}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@config_options
@click.option(
    "--comparison", "-C", default="noder",
    help="Comparison type (noder/remind/ablation/avg_impute/forward_impute)",
)
@click.option("--run-name", default=None, help="Explicit run name")
def evaluate(config_path, fold, overrides, comparison, run_name):
    """Run evaluation/comparison analysis."""
    cfg = load_config(config_path, list(overrides), fold)
    if run_name:
        cfg = _with_run_name(cfg, run_name)

    from synbadd.evaluation import build_evaluator

    evaluator = build_evaluator(comparison, cfg)
    results = evaluator.report()
    if cfg.wandb.enabled and not cfg.system.debug:
        evaluator.log_wandb(results)
    click.echo(f"Evaluation ({comparison}) complete for fold {cfg.data.fold}.")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

@main.command()
@config_options
@click.option("--folds", default="1,2,3,4,5", help="Comma-separated fold indices")
@click.option("--stages", default="train,infer,evaluate", help="Comma-separated stages to run")
@click.option("--skip-existing", is_flag=True, help="Skip stages whose outputs already exist")
@click.option("--comparison", "-C", default="noder", help="Evaluation comparison type")
@click.option(
    "--infer-method", default="synbadd",
    type=click.Choice(["synbadd", "avg_impute", "forward_impute"]),
    help="Inference method for the infer stage",
)
def pipeline(config_path, fold, overrides, folds, stages, skip_existing, comparison, infer_method):
    """Run full train -> infer -> evaluate pipeline."""
    fold_list = [int(f) for f in folds.split(",")]
    stage_list = [s.strip() for s in stages.split(",")]

    cfg = load_config(config_path, list(overrides))
    from synbadd.pipeline.orchestrator import Pipeline

    p = Pipeline(cfg, fold_list, stage_list, comparison_type=comparison, infer_method=infer_method)
    p.run(skip_existing=skip_existing)
    click.echo("Pipeline complete.")


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

@main.group()
def experiment():
    """Run robustness experiments on a pre-trained model."""
    pass


@experiment.command("list")
def experiment_list():
    """List all registered experiments."""
    import synbadd.experiments  # noqa: F401  (triggers registry)
    from synbadd.experiments.registry import list_experiments
    for name in list_experiments():
        click.echo(name)


@experiment.command("run")
@config_options
@click.option("--experiment-name", "-e", required=True,
              help="Registered experiment name (see `synbadd experiment list`)")
@click.option("--run-name", required=True,
              help="Pre-trained run name (filename stem of the checkpoint)")
@click.option("--skip-inference", is_flag=True,
              help="Skip the inference phase; aggregate from existing per_config_metrics.csv")
@click.option("--skip-existing", is_flag=True,
              help="Skip individual configs whose .nii.gz already exists")
@click.option("--recover-metrics", is_flag=True,
              help="Skip inference; recompute metrics from saved .nii.gz files (no model)")
@click.option("--with-dice", is_flag=True,
              help="Compute Dice + volume per ROI from saved .nii.gz (uses ANTs SyN; slow)")
@click.option("--cleanup", is_flag=True,
              help="Delete .nii.gz files after aggregation; keeps CSVs")
def experiment_run(config_path, fold, overrides, experiment_name, run_name,
                   skip_inference, skip_existing, recover_metrics, with_dice, cleanup):
    """Run a robustness experiment on a pre-trained model."""
    cfg = load_config(config_path, list(overrides), fold)
    cfg = _with_run_name(cfg, run_name)

    import synbadd.experiments  # noqa: F401
    from synbadd.data.loader import _DATASET_CLASSES
    from synbadd.experiments.registry import build_experiment
    from synbadd.experiments.runner import ExperimentRunner

    dataset_cls = _DATASET_CLASSES.get(cfg.data.svf_type)
    if dataset_cls is None:
        raise click.ClickException(
            f"Unknown svf_type '{cfg.data.svf_type}'. "
            f"Available: {list(_DATASET_CLASSES)}"
        )
    dataset = dataset_cls(cfg, cfg.paths.csv_file, cfg.data.fold, subset="test")

    exp = build_experiment(experiment_name, cfg, cfg.data.fold)
    runner = ExperimentRunner(cfg, exp, dataset)
    runner.run(
        skip_inference=skip_inference,
        skip_existing=skip_existing,
        recover_metrics=recover_metrics,
        with_dice=with_dice,
        cleanup=cleanup,
    )
    click.echo(
        f"Experiment '{experiment_name}' complete (fold {cfg.data.fold}). "
        f"Outputs: {runner.output_dir}"
    )


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

@main.group()
def manifest():
    """Query experiment manifests."""
    pass


@manifest.command("query")
@click.option("--manifest-dir", required=True, help="Directory containing manifest JSON files")
@click.option("--fold", type=int, default=None, help="Filter by fold")
@click.option("--metric", default="best_val_loss", help="Sort by this metric")
@click.option("--top", type=int, default=5, help="Show top N results")
def manifest_query(manifest_dir, fold, metric, top):
    """Query manifests for best models."""
    from synbadd.pipeline.manifest import ManifestStore

    store = ManifestStore(manifest_dir)
    results = store.query(fold=fold, metric=metric, top=top)
    if not results:
        click.echo("No matching runs found.")
        return
    click.echo(f"{'Run Name':<60} {metric}")
    click.echo("-" * 80)
    for run_name, value in results:
        click.echo(f"{run_name:<60} {value}")


@manifest.command("show")
@click.argument("run_name")
@click.option("--manifest-dir", required=True, help="Directory containing manifest JSON files")
def manifest_show(run_name, manifest_dir):
    """Show full manifest for a specific run."""
    import json
    import os

    path = os.path.join(manifest_dir, f"{run_name}.json")
    if not os.path.exists(path):
        click.echo(f"No manifest found for: {run_name}")
        raise SystemExit(1)
    with open(path) as f:
        data = json.load(f)
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _with_run_name(cfg, run_name):
    """Return a new config with the run_name set."""
    import dataclasses
    return dataclasses.replace(cfg, run_name=run_name)
