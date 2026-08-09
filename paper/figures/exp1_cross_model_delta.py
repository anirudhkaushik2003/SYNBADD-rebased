#!/usr/bin/env python3
"""Exp 1 -- Cross-model paired-delta bar chart with significance markers.

Mean per-subject change in metric when 2 inputs are dropped, paired across
the same 76 subjects (cohort-stable n=0 -> n=2).

Run:
    python paper/figures/exp1_cross_model_delta.py
Output:
    paper/images/exp1_cross_model_delta.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import (MODEL_COLOR, MODEL_LABEL, OUTPUT_ROOT, apply_style,
                    save_pdf)


SIG_CSV = OUTPUT_ROOT / "significance_tests_fold2" / "significance_tests.csv"


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

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.7))
    axes = axes.ravel()
    w = 0.36

    main_d = cm["model_a_mean_delta"].values
    nu_d   = cm["model_b_mean_delta"].values

    for i, (ax, metric, label, main_value, nounet_value) in enumerate(
            zip(axes, [m[0] for m in metrics], [m[1] for m in metrics],
                main_d, nu_d)):
        values = [main_value, nounet_value]
        bars1 = ax.bar([-0.5], [main_value], w, color=MODEL_COLOR["main"],
                       label=MODEL_LABEL["main"], edgecolor="black", lw=0.4)
        bars2 = ax.bar([0.5], [nounet_value], w, color=MODEL_COLOR["nounet"],
                       label=MODEL_LABEL["nounet"], edgecolor="black", lw=0.4)
        magnitude = max(abs(value) for value in values)
        pad = max(magnitude * 0.18, 1e-4)
        for xpos, value in zip([-0.5, 0.5], values):
            offset = pad if value >= 0 else -pad
            ax.text(xpos, value + offset, f"{value:.3g}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([-0.5, 0.5])
        ax.set_xticklabels(["Full", "No refinement"])
        ax.set_ylim(min(min(values) - pad * 3, -pad), max(values) + pad * 4)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel(r"Mean per-subject $\Delta$")
    axes[2].set_ylabel(r"Mean per-subject $\Delta$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out = save_pdf(fig, "exp1_cross_model_delta")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
