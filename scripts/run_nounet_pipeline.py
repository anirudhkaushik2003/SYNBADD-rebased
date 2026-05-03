#!/usr/bin/env python3
"""
NoUnet+NoDAFT pipeline: inference → evaluation → cleanup, fold by fold.

For each fold:
  1. Run inference (saves synthetic .nii.gz to SYNBADD_Qualitative_{N}/{run_name}/)
  2. Compute Dice/volume/SSIM/PSNR and log to W&B
  3. Delete the inference output directory to free disk space
  4. Move to next fold

Usage:
    python scripts/run_nounet_pipeline.py [--folds 1,2,3,4,5] [--skip-cleanup]
"""

import argparse
import os
import shutil

CONFIGS_DIR = "/root/anirudhkaushik/SYNBADD-rebased/configs/experiments"
PROJECT = "SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_UnetAblations"
OUTPUT_BASE = "/scratch/anirudhkaushik"

NOUNET_RUNS = {
    1: "SYNBADD_128_6_5e-05_0.0_2025-05-28_10-18-38_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold1",
    2: "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2",
    3: "SYNBADD_128_6_5e-05_0.0_2025-05-28_21-02-34_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold3",
    4: "SYNBADD_128_6_5e-05_0.0_2025-05-29_02-52-23_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold4",
    5: "SYNBADD_128_6_5e-05_0.0_2025-05-29_09-56-27_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold5",
}

# Fold 5 NoUnet checkpoint was accidentally saved with using_unet=True architecture.
# Must build model with using_unet=True so the checkpoint loads cleanly.
# The generator forward will still use the NoUnet (projection) path at runtime.
FOLD5_USES_UNET_ARCH = True


def run_inference(cfg, fold, run_name):
    from synbadd.data.loader import build_dataloaders
    from synbadd.engine.inferrer import build_inferrer

    print(f"\n  >> Running inference for NoUnet+NoDAFT fold {fold}...")
    cfg.run_name = run_name
    loaders = build_dataloaders(cfg, fold=fold)
    test_loader = loaders["test"]
    inferrer = build_inferrer(cfg, test_loader=test_loader, method="synbadd")
    result = inferrer.run()
    return result["output_dir"]


def run_evaluation(cfg, fold, run_name):
    from synbadd.evaluation.ablation_comparison import AblationComparison

    print(f"\n  >> Running Dice/volume evaluation for NoUnet+NoDAFT fold {fold}...")
    evaluator = AblationComparison(
        cfg, fold=fold,
        run_names=[run_name],
        variant_labels=["NoUnet+NoDAFT"],
    )
    results = evaluator.report()
    evaluator.log_wandb(results)

    summary = results.get("summary_metrics")
    if summary is not None and not summary.empty:
        print(f"\n{summary.to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Keep inference outputs after evaluation")
    args = parser.parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    from synbadd.config.loader import load_config

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  NoUnet+NoDAFT — Fold {fold}")
        print(f"{'='*60}")

        run_name = NOUNET_RUNS[fold]
        output_dir = os.path.join(OUTPUT_BASE, f"SYNBADD_Qualitative_{fold}", run_name)

        # Fold 5 checkpoint was saved with using_unet=True — match the architecture
        if fold == 5 and FOLD5_USES_UNET_ARCH:
            print("  NOTE: Fold 5 checkpoint has OnlyUnet architecture. Loading with using_unet=True.")
            cfg = load_config(
                f"{CONFIGS_DIR}/unet_ablation_onlyunet.yaml",
                overrides=[f"project_name={PROJECT}"],
                fold=fold,
            )
        else:
            cfg = load_config(
                f"{CONFIGS_DIR}/unet_ablation_nounet.yaml",
                overrides=[f"project_name={PROJECT}"],
                fold=fold,
            )

        # Step 1: Inference (skip if already done)
        if os.path.isdir(output_dir) and any(
            f.endswith("_synthetic.nii.gz") for f in os.listdir(output_dir)
        ):
            print(f"  Inference outputs already exist at {output_dir}, skipping inference.")
        else:
            run_inference(cfg, fold, run_name)

        # Step 2: Evaluation
        run_evaluation(cfg, fold, run_name)

        # Step 3: Cleanup
        if not args.skip_cleanup:
            if os.path.isdir(output_dir):
                print(f"\n  >> Cleaning up inference outputs: {output_dir}")
                shutil.rmtree(output_dir)
                print(f"  Deleted {output_dir}")
        else:
            print(f"  Skipping cleanup (--skip-cleanup set).")

    print(f"\n{'='*60}")
    print("  ALL FOLDS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
