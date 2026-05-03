#!/usr/bin/env python3
"""
Interactive SYNBADD inference + evaluation pipeline.

Use arrow keys to select:
  - Model type (temporal architecture)
  - Modifier (DAFT / OnlyUnet / NoUnet / none)
  - Direction mode (Past+Future, OnlyPast, …)
  - Training run date (if multiple exist)
  - Folds to process
  - Pipeline mode (inference+eval / inference only / eval only)

Runs fold-by-fold: inference → eval → cleanup (to save disk).

Usage:
    python scripts/run_pipeline.py [--ckpt-root DIR] [--output-root DIR] [--list-only]
"""

import argparse
import os
import shutil
import sys

import questionary
from questionary import Style

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CKPT_ROOT       = "/scratch/anirudhkaushik/SYNBADD"
OUTPUT_ROOT     = "/scratch/anirudhkaushik"
CONFIGS_DIR     = "/root/anirudhkaushik/SYNBADD-rebased/configs/experiments"
BASE_CONFIG     = f"{CONFIGS_DIR}/../base.yaml"

STYLE = Style([
    ("qmark",        "fg:#00bfff bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00ff87 bold"),
    ("pointer",      "fg:#ff8c00 bold"),
    ("highlighted",  "fg:#ff8c00 bold"),
    ("selected",     "fg:#00ff87"),
    ("separator",    "fg:#555555"),
    ("instruction",  "fg:#888888"),
])


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
MODIFIER_CONFIG = {
    "DAFT":     {"model.generator.using_daft": "true",  "model.generator.using_unet": "true"},
    "OnlyUnet": {"model.generator.using_daft": "false", "model.generator.using_unet": "true"},
    "NoUnet":   {"model.generator.using_daft": "false", "model.generator.using_unet": "false"},
    None:       {"model.generator.using_daft": "true",  "model.generator.using_unet": "true"},
}

# Map run-name temporal string → config YAML value
TEMPORAL_TO_CONFIG = {
    "GatedCrossAttentionTransformer": "GatedCrossAttentionTransformer",
    "CrossAttentionGatedTransformer": "CrossAttentionGatedTransformer",
    "CrossAttentionTransformer":      "CrossAttentionTransformer",
    "AveragedImputation":             "AveragedImputation",
}


def build_cfg(run_info, fold: int, project: str):
    """Build a SynbaddConfig for the given run and fold."""
    from synbadd.config.loader import load_config
    from synbadd.discovery import MODIFIER_CONFIG as _MC

    temporal_type = TEMPORAL_TO_CONFIG.get(run_info.temporal, run_info.temporal)
    mod_cfg = MODIFIER_CONFIG.get(run_info.modifier, MODIFIER_CONFIG[None])

    overrides = [
        f"project_name={project}",
        f"model.temporal.type={temporal_type}",
        f"model.discriminator.enabled=false",
    ] + [f"{k}={v}" for k, v in mod_cfg.items()]

    cfg = load_config(BASE_CONFIG, overrides=overrides, fold=fold)
    cfg.run_name = run_info.run_name
    cfg.paths.output_dir = OUTPUT_ROOT
    return cfg


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_inference(cfg, fold: int, run_name: str):
    from synbadd.data.loader import build_dataloaders
    from synbadd.engine.inferrer import build_inferrer

    print(f"\n  [inference] fold {fold}  run={run_name}")
    loaders = build_dataloaders(cfg, fold=fold)
    test_loader = loaders["test"]
    cfg.run_name = run_name
    inferrer = build_inferrer(cfg, test_loader=test_loader, method="synbadd")
    result = inferrer.run()
    return result["output_dir"]


def run_evaluation(cfg, fold: int, run_name: str, label: str, project: str):
    from synbadd.evaluation.ablation_comparison import AblationComparison

    print(f"\n  [evaluation] fold {fold}  label={label}")
    evaluator = AblationComparison(
        cfg, fold=fold,
        run_names=[run_name],
        variant_labels=[label],
    )
    results = evaluator.report()
    evaluator.log_wandb(results)

    summary = results.get("summary_metrics")
    if summary is not None and not summary.empty:
        print(f"\n{summary.to_string(index=False)}")


def cleanup_output(output_dir: str):
    if os.path.isdir(output_dir):
        print(f"\n  [cleanup] deleting {output_dir}")
        shutil.rmtree(output_dir)


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------

def select(prompt: str, choices, multi=False, instruction=""):
    """Wrapper around questionary select / checkbox."""
    kwargs = dict(style=STYLE, instruction=instruction)
    if multi:
        return questionary.checkbox(prompt, choices=choices, **kwargs).ask()
    return questionary.select(prompt, choices=choices, **kwargs).ask()


def run_interactive(catalog, args):
    from synbadd.discovery import RunInfo

    # ---- Step 1: temporal type ----
    temporals = sorted({k[0] for k in catalog})
    temporal = select("Model type (temporal architecture):", temporals)
    if temporal is None:
        sys.exit(0)

    # ---- Step 2: modifier ----
    modifiers = sorted({k[1] for k in catalog if k[0] == temporal}, key=lambda x: x or "")
    mod_labels = [m if m else "(none / default DAFT)" for m in modifiers]
    mod_label = select("Variant (modifier):", mod_labels)
    if mod_label is None:
        sys.exit(0)
    modifier = None if mod_label.startswith("(none") else mod_label

    # ---- Step 3: direction mode ----
    modes = sorted({k[2] for k in catalog if k[0] == temporal and k[1] == modifier})
    if len(modes) == 1:
        mode = modes[0]
        print(f"  Direction mode: {mode}  (only option)")
    else:
        mode = select("Direction mode:", modes)
        if mode is None:
            sys.exit(0)

    # ---- Get matching runs ----
    key = (temporal, modifier, mode)
    runs = catalog.get(key, [])
    if not runs:
        print("No runs found for this combination. Exiting.")
        sys.exit(1)

    # ---- Step 4: project (if multiple) ----
    projects = sorted({r.project for r in runs if r.project})
    if len(projects) > 1:
        project = select("Training project / experiment group:", projects)
        if project is None:
            sys.exit(0)
        runs = [r for r in runs if r.project == project]
    elif projects:
        project = projects[0]
        print(f"  Project: {project}")
    else:
        project = ""

    # ---- Step 5: per-fold run selection ----
    # Group runs by fold, show date options per fold (tiebreak only when needed)
    available_folds = sorted({r.fold for r in runs})
    fold_choices = []
    for f in available_folds:
        fold_runs = sorted([r for r in runs if r.fold == f], key=lambda r: (r.date, r.time))
        if not fold_runs:
            continue
        # Show all options for this fold as a single choice entry
        parts = []
        for r in fold_runs:
            status = []
            if r.has_checkpoint:
                status.append("ckpt ✓")
            if r.inference_done:
                status.append(f"infer done ({r.n_synthetics})")
            parts.append(f"{r.datetime_str} [{', '.join(status) or 'no ckpt/infer'}]")
        label = f"Fold {f}:  " + "  |  ".join(parts)
        fold_choices.append(questionary.Choice(title=label, value=f))

    chosen_folds = select("Select folds to run:", fold_choices, multi=True,
                          instruction="(space to select, enter to confirm)")
    if not chosen_folds:
        print("No folds selected. Exiting.")
        sys.exit(0)

    # For each chosen fold, resolve which run to use (prompt only if truly ambiguous)
    fold_map = {}
    for f in chosen_folds:
        fold_runs = sorted([r for r in runs if r.fold == f], key=lambda r: (r.date, r.time))
        if len(fold_runs) == 1:
            fold_map[f] = fold_runs[0]
        else:
            date_choices = []
            for r in fold_runs:
                status = []
                if r.has_checkpoint:
                    status.append("ckpt ✓")
                if r.inference_done:
                    status.append(f"infer done ({r.n_synthetics} scans)")
                lbl = f"{r.datetime_str}  [{', '.join(status) or 'no ckpt/infer'}]"
                date_choices.append(questionary.Choice(title=lbl, value=r))
            chosen_run = select(f"Fold {f} — multiple runs found, pick one:", date_choices)
            if chosen_run is None:
                sys.exit(0)
            fold_map[f] = chosen_run

    # ---- Step 7: pipeline mode ----
    pipeline_mode = select("Pipeline mode:", [
        questionary.Choice("inference + evaluation (cleanup after each fold)", "infer_eval"),
        questionary.Choice("inference only", "infer"),
        questionary.Choice("evaluation only (inference outputs must exist)", "eval"),
        questionary.Choice("inference + evaluation (keep outputs)", "infer_eval_keep"),
    ])
    if pipeline_mode is None:
        sys.exit(0)

    do_infer  = "infer" in pipeline_mode
    do_eval   = "eval"  in pipeline_mode
    do_cleanup = pipeline_mode == "infer_eval"

    # ---- Confirm ----
    label = f"{temporal}+{modifier}" if modifier else temporal
    print(f"\n{'='*60}")
    print(f"  Model    : {label}  [{mode}]")
    print(f"  Project  : {project}")
    print(f"  Folds    : {chosen_folds}")
    print(f"  Infer    : {do_infer}   Eval: {do_eval}   Cleanup: {do_cleanup}")
    print(f"{'='*60}")
    ok = questionary.confirm("Proceed?", default=True, style=STYLE).ask()
    if not ok:
        sys.exit(0)

    # ---- Run ----
    for fold in chosen_folds:
        run_info = fold_map.get(fold)
        if run_info is None:
            print(f"  No run info for fold {fold}, skipping.")
            continue

        run_name = run_info.run_name
        output_dir = os.path.join(OUTPUT_ROOT, f"SYNBADD_Qualitative_{fold}", run_name)

        print(f"\n{'='*60}")
        print(f"  {label}  —  Fold {fold}  ({run_info.datetime_str})")
        print(f"{'='*60}")

        cfg = build_cfg(run_info, fold, project)

        if do_infer:
            if os.path.isdir(output_dir) and any(
                f.endswith("_synthetic.nii.gz") for f in os.listdir(output_dir)
            ):
                print(f"  Inference already done ({output_dir}), skipping.")
            else:
                run_inference(cfg, fold, run_name)

        if do_eval:
            run_evaluation(cfg, fold, run_name, label=label, project=project)

        if do_cleanup and do_infer:
            cleanup_output(output_dir)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive SYNBADD pipeline runner")
    parser.add_argument("--ckpt-root",    default=CKPT_ROOT,   help="Checkpoint root directory")
    parser.add_argument("--output-root",  default=OUTPUT_ROOT, help="Output (qualitative) root")
    parser.add_argument("--list-only",    action="store_true",  help="Print catalog and exit")
    args = parser.parse_args()

    from synbadd.discovery import build_catalog, print_catalog

    print("Scanning checkpoints and inference outputs...")
    catalog = build_catalog(
        ckpt_root=args.ckpt_root,
        qualitative_base=args.output_root,
    )

    if args.list_only or not catalog:
        print_catalog(catalog)
        return

    print_catalog(catalog)
    run_interactive(catalog, args)


if __name__ == "__main__":
    main()
