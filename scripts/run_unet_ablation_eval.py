#!/usr/bin/env python3
"""Run inference + evaluation (Dice/volume) for UNet ablation models across all 5 folds.

Usage:
    python scripts/run_unet_ablation_eval.py [--folds 1,2,3,4,5] [--skip-inference]
"""

import argparse
import dataclasses

from synbadd.config.loader import load_config
from synbadd.data.loader import build_dataloaders
from synbadd.engine.inferrer import build_inferrer
from synbadd.evaluation.ablation_comparison import AblationComparison

PROJECT = "SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_UnetAblations"
CONFIGS_DIR = "/root/anirudhkaushik/SYNBADD-rebased/configs/experiments"

# Run names per fold
ONLYUNET_RUNS = {
    1: "SYNBADD_128_6_5e-05_0.0_2025-05-26_16-21-58_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1",
    2: "SYNBADD_128_6_5e-05_0.0_2025-05-26_23-38-04_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold2",
    3: "SYNBADD_128_6_5e-05_0.0_2025-05-27_08-29-07_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold3",
    4: "SYNBADD_128_6_5e-05_0.0_2025-05-27_14-41-32_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold4",
    5: "SYNBADD_128_6_5e-05_0.0_2025-05-27_21-54-06_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold5",
}

NOUNET_RUNS = {
    1: "SYNBADD_128_6_5e-05_0.0_2025-05-28_10-18-38_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold1",
    2: "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2",
    3: "SYNBADD_128_6_5e-05_0.0_2025-05-28_21-02-34_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold3",
    4: "SYNBADD_128_6_5e-05_0.0_2025-05-29_02-52-23_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold4",
    5: "SYNBADD_128_6_5e-05_0.0_2025-05-29_09-56-27_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold5",
}


def run_inference(fold, config_path, run_name):
    """Run inference for a single variant/fold."""
    cfg = load_config(config_path, overrides=[f"project_name={PROJECT}"], fold=fold)
    cfg = dataclasses.replace(cfg, run_name=run_name)

    loaders = build_dataloaders(cfg, fold)
    inferrer = build_inferrer(cfg, test_loader=loaders["test"], method="synbadd")
    result = inferrer.run()
    print(f"  -> {result['n_subjects']} subjects, outputs: {result['output_dir']}")
    return result


def run_evaluation(fold, run_names, variant_labels):
    """Run Dice/volume evaluation comparing variants for a fold."""
    cfg = load_config(f"{CONFIGS_DIR}/unet_ablation_onlyunet.yaml",
                      overrides=[f"project_name={PROJECT}"], fold=fold)

    evaluator = AblationComparison(
        cfg, fold=fold,
        run_names=run_names,
        variant_labels=variant_labels,
    )
    results = evaluator.report()
    evaluator.log_wandb(results)
    print(f"  -> Evaluation complete for fold {fold}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference, only run evaluation (assumes .nii.gz already exist)")
    args = parser.parse_args()

    folds = [int(f) for f in args.folds.split(",")]

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}")
        print(f"{'='*60}")

        if not args.skip_inference:
            # OnlyUnet inference
            print(f"\n--- OnlyUnet+NoDAFT inference (fold {fold}) ---")
            run_inference(fold, f"{CONFIGS_DIR}/unet_ablation_onlyunet.yaml", ONLYUNET_RUNS[fold])

            # NoUnet inference
            print(f"\n--- NoUnet+NoDAFT inference (fold {fold}) ---")
            run_inference(fold, f"{CONFIGS_DIR}/unet_ablation_nounet.yaml", NOUNET_RUNS[fold])

        # Evaluation: compare both variants
        print(f"\n--- Dice/Volume evaluation (fold {fold}) ---")
        run_evaluation(
            fold,
            run_names=[ONLYUNET_RUNS[fold], NOUNET_RUNS[fold]],
            variant_labels=["OnlyUnet+NoDAFT", "NoUnet+NoDAFT"],
        )

    print(f"\n{'='*60}")
    print("  ALL FOLDS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
