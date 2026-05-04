#!/usr/bin/env bash
# Run robustness experiments on a chosen pretrained model (fold 2).
#
# Usage:
#   ./scripts/run_robustness_experiments.sh                       # main model, both experiments
#   ./scripts/run_robustness_experiments.sh nounet                # NoUnet+NoDAFT model, both experiments
#   ./scripts/run_robustness_experiments.sh main exp1             # main, only exp 1
#   ./scripts/run_robustness_experiments.sh nounet exp1 --skip-inference
#
# Positional args:
#   $1 (model):    main | nounet     (default: main)
#   $2 (which):    both | exp1 | exp2  (default: both)
# Remaining args are forwarded to `synbadd experiment run` as-is.
#
# Env:
#   NO_DICE=1   skip Dice/volume computation
set -euo pipefail

# Activate the conda env that has synbadd installed
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

MODEL="${1:-main}"
WHICH="${2:-both}"
shift $(( $# >= 2 ? 2 : $# ))

case "${MODEL}" in
    main)
        CONFIG="configs/experiments/cross_attn_DAFT_PF.yaml"
        RUN_NAME="SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2"
        LABEL="CrossAttentionTransformer + UNet + DAFT"
        ;;
    nounet|NoUnet|NoUnetNoDAFT)
        CONFIG="configs/experiments/cross_attn_NoUnetNoDAFT_PF.yaml"
        RUN_NAME="SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2"
        LABEL="CrossAttentionTransformer + NoUnet + NoDAFT"
        ;;
    *)
        echo "Unknown model '${MODEL}'. Choose: main | nounet" >&2
        exit 1
        ;;
esac

DICE_FLAG="--with-dice"
[[ "${NO_DICE:-0}" == "1" ]] && DICE_FLAG=""
EXTRA_FLAGS="${DICE_FLAG} $@"

run_one() {
    local exp="$1"
    echo "============================================================"
    echo " ${LABEL}"
    echo " Experiment: ${exp}  (fold 2)"
    echo "============================================================"
    synbadd experiment run \
        -c "${CONFIG}" \
        -e "${exp}" \
        --run-name "${RUN_NAME}" \
        ${EXTRA_FLAGS}
    echo
}

case "${WHICH}" in
    both)
        run_one reduced_time_input
        run_one asymmetric_information
        ;;
    exp1|reduced|reduced_time_input)
        run_one reduced_time_input
        ;;
    exp2|asymmetric|asymmetric_information)
        run_one asymmetric_information
        ;;
    *)
        echo "Unknown experiment selection '${WHICH}'. Choose: both | exp1 | exp2" >&2
        exit 1
        ;;
esac

echo "Done — model='${MODEL}', experiments='${WHICH}'."
