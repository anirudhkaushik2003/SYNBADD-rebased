#!/bin/bash
# Run inference for Unet ablation models (OnlyUnet+NoDAFT and NoUnet+NoDAFT) across all 5 folds.
set -e

CONFIGS_DIR="/root/anirudhkaushik/SYNBADD-rebased/configs/experiments"
PROJECT="SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_UnetAblations"

# OnlyUnet run names per fold
ONLYUNET_RUNS=(
  "SYNBADD_128_6_5e-05_0.0_2025-05-26_16-21-58_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1"
  "SYNBADD_128_6_5e-05_0.0_2025-05-26_23-38-04_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold2"
  "SYNBADD_128_6_5e-05_0.0_2025-05-27_08-29-07_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold3"
  "SYNBADD_128_6_5e-05_0.0_2025-05-27_14-41-32_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold4"
  "SYNBADD_128_6_5e-05_0.0_2025-05-27_21-54-06_ADNI_all_longitudinal_CrossAttentionTransformer+OnlyUnet_Past+Future_Fold5"
)

# NoUnet run names per fold
NOUNET_RUNS=(
  "SYNBADD_128_6_5e-05_0.0_2025-05-28_10-18-38_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold1"
  "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2"
  "SYNBADD_128_6_5e-05_0.0_2025-05-28_21-02-34_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold3"
  "SYNBADD_128_6_5e-05_0.0_2025-05-29_02-52-23_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold4"
  "SYNBADD_128_6_5e-05_0.0_2025-05-29_09-56-27_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold5"
)

echo "=========================================="
echo "  OnlyUnet+NoDAFT Inference (5 folds)"
echo "=========================================="
for FOLD in 1 2 3 4 5; do
  IDX=$((FOLD - 1))
  RUN_NAME="${ONLYUNET_RUNS[$IDX]}"
  echo ""
  echo ">>> Fold $FOLD: $RUN_NAME"
  synbadd infer \
    -c "$CONFIGS_DIR/unet_ablation_onlyunet.yaml" \
    --fold "$FOLD" \
    --run-name "$RUN_NAME" \
    --set "project_name=$PROJECT"
done

echo ""
echo "=========================================="
echo "  NoUnet+NoDAFT Inference (5 folds)"
echo "=========================================="
for FOLD in 1 2 3 4 5; do
  IDX=$((FOLD - 1))
  RUN_NAME="${NOUNET_RUNS[$IDX]}"
  echo ""
  echo ">>> Fold $FOLD: $RUN_NAME"
  synbadd infer \
    -c "$CONFIGS_DIR/unet_ablation_nounet.yaml" \
    --fold "$FOLD" \
    --run-name "$RUN_NAME" \
    --set "project_name=$PROJECT"
done

echo ""
echo "=========================================="
echo "  All inference complete!"
echo "=========================================="
