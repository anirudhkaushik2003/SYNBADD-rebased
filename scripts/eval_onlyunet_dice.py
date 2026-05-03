#!/usr/bin/env python3
"""Run Dice/volume evaluation for OnlyUnet+NoDAFT across all 5 folds.

Usage:
    python scripts/eval_onlyunet_dice.py [--folds 1,2,3,4,5]
"""

import argparse

from synbadd.config.loader import load_config
from synbadd.evaluation.ablation_comparison import AblationComparison

CONFIGS_DIR = "/root/anirudhkaushik/SYNBADD-rebased/configs/experiments"
PROJECT = "SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_UnetAblations"

ONLYUNET_RUNS = {
    1: "SYNBADD_128_6_5e-05_0.0_2025-05-26_16-21-58_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1",
    2: "SYNBADD_128_6_5e-05_0.0_2025-05-26_23-38-04_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold2",
    3: "SYNBADD_128_6_5e-05_0.0_2025-05-27_08-29-07_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold3",
    4: "SYNBADD_128_6_5e-05_0.0_2025-05-27_14-41-32_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold4",
    5: "SYNBADD_128_6_5e-05_0.0_2025-05-27_21-54-06_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold5",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="1,2,3,4,5")
    args = parser.parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  Dice/Volume evaluation — OnlyUnet+NoDAFT — Fold {fold}")
        print(f"{'='*60}")

        cfg = load_config(
            f"{CONFIGS_DIR}/unet_ablation_onlyunet.yaml",
            overrides=[f"project_name={PROJECT}"],
            fold=fold,
        )

        evaluator = AblationComparison(
            cfg, fold=fold,
            run_names=[ONLYUNET_RUNS[fold]],
            variant_labels=["OnlyUnet+NoDAFT"],
        )
        results = evaluator.report()
        evaluator.log_wandb(results)

        # Print summary
        summary = results.get("summary_metrics")
        if summary is not None and not summary.empty:
            print(f"\n{summary.to_string(index=False)}")

    print(f"\n{'='*60}")
    print("  ALL FOLDS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
