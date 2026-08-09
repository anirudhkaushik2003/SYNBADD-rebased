#!/usr/bin/env python3
"""Exp 1 -- Average Dice across 8 ROIs vs n_dropped, per group, both models.

Run:
    python paper/figures/exp1_dice_curves.py
Output:
    paper/images/exp1_dice_curves.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from _style import GROUP_COLOR, MODEL_LABEL, apply_style, load, save_pdf


def main():
    apply_style()
    _, dice_main = load("main", "exp1")
    _, dice_nu   = load("nounet", "exp1")

    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.0), sharex=True, sharey=True)
    models = [(dice_main, "main"), (dice_nu, "nounet")]
    groups = ["CN", "MCI", "AD"]

    for col, (dice, model) in enumerate(models):
        # Per-subject average over ROIs, then summarize each diagnostic group.
        avg = (dice.groupby(["subject", "group", "n_dropped"])["dice"].mean()
                   .reset_index())
        for row, grp in enumerate(groups):
            ax = axes[row, col]
            sub = avg[avg.group == grp]
            if not sub.empty:
                mean = sub.groupby("n_dropped")["dice"].mean().sort_index()
                count = sub.groupby("n_dropped")["dice"].count().sort_index()
                sem = (sub.groupby("n_dropped")["dice"].std() /
                       np.sqrt(count)).sort_index().fillna(0)
                ax.errorbar(
                    mean.index, mean.values, yerr=sem.values,
                    fmt="o-", color=GROUP_COLOR[grp], capsize=2, lw=1.2, ms=4,
                )
            ax.set_xticks(range(0, 5))
            ax.text(0.04, 0.92, grp, transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, fontweight="bold")
            ax.grid(True, axis="both", alpha=0.22)

        axes[0, col].set_title(MODEL_LABEL[model], fontsize=11, pad=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Dice")
    for ax in axes[-1, :]:
        ax.set_xlabel("Inputs dropped")
    fig.text(0.02, 0.50, "Mean Dice (average over 8 ROIs)", rotation="vertical",
             va="center", ha="center")
    fig.tight_layout(rect=(0.07, 0.03, 1, 0.96), w_pad=1.0, h_pad=1.2)
    out = save_pdf(fig, "exp1_dice_curves")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
