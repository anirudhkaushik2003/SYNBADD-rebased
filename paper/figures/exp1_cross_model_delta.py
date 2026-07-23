#!/usr/bin/env python3
"""Exp 1 -- Cross-model paired-delta bar chart with significance markers.

Mean per-subject change in metric when 2 inputs are dropped, paired across
the same 76 subjects (cohort-stable n=0 -> n=2).

Run:
    python paper/figures/exp1_cross_model_delta.py
Output:
    paper/images/exp1_cross_model_delta.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import MODEL_COLOR, MODEL_LABEL, apply_style, save_pdf


SIG_CSV = Path("/scratch/anirudhkaushik/SYNBADD_outputs/significance_tests_fold2/significance_tests.csv")


def main():
    apply_style()
    df = pd.read_csv(SIG_CSV)
    cm = df[(df.test == "cross_model_delta_wilcoxon") &
            (df.experiment == "reduced_time_input") &
            (df.scope == "all_cohort_stable_n0_n2")]

    metrics = [("ssim", "SSIM"),
               ("psnr", "PSNR (dB)"),
               ("epe",  "EPE (voxels)"),
               ("dice_avg_roi", "Dice (avg ROIs)")]
    cm = cm.set_index("metric").reindex([m[0] for m in metrics])

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(metrics))
    w = 0.36

    main_d = cm["model_a_mean_delta"].values
    nu_d   = cm["model_b_mean_delta"].values

    bars1 = ax.bar(x - w/2, main_d, w, color=MODEL_COLOR["main"],
                   label=MODEL_LABEL["main"], edgecolor="black", lw=0.4)
    bars2 = ax.bar(x + w/2, nu_d, w, color=MODEL_COLOR["nounet"],
                   label=MODEL_LABEL["nounet"], edgecolor="black", lw=0.4)

    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        p = cm["p_value"].iloc[i]
        if pd.isna(p):
            continue
        if p < 0.001:   marker = r"$***$"
        elif p < 0.01:  marker = r"$**$"
        elif p < 0.05:  marker = r"$*$"
        else:           marker = "ns"
        top = max(abs(b1.get_height()), abs(b2.get_height()))
        ax.text(i, top * 1.18 if top > 0 else 0.0005, marker,
                ha="center", va="bottom", fontsize=10)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel(r"Mean per-subject $\Delta$ ($n_{\mathrm{drop}}{=}0 \rightarrow 2$)")
    ax.set_title(r"Exp.~1 -- Cross-model paired-$\Delta$ (cohort-stable, $n{=}76$)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = save_pdf(fig, "exp1_cross_model_delta")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
