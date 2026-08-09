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

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True, sharey=True)
    models = [(dice_main, "main"), (dice_nu, "nounet")]

    for ax, (dice, model) in zip(axes, models):
        # Per-subject avg over ROIs, then group by (subject, group, n_dropped)
        avg = (dice.groupby(["subject", "group", "n_dropped"])["dice"].mean()
                   .reset_index())
        for grp in ["CN", "MCI", "AD"]:
            sub = avg[avg.group == grp]
            if sub.empty:
                continue
            mean = sub.groupby("n_dropped")["dice"].mean().sort_index()
            count = sub.groupby("n_dropped")["dice"].count().sort_index()
            sem = (sub.groupby("n_dropped")["dice"].std() / np.sqrt(count)).sort_index()
            ax.errorbar(
                mean.index, mean.values, yerr=sem.values,
                fmt="o-", color=GROUP_COLOR[grp], capsize=2, lw=1.0, ms=4,
                label=grp,
            )
        ax.set_xticks(range(0, 5))
        ax.text(0.03, 0.95, MODEL_LABEL[model], transform=ax.transAxes,
                ha="left", va="top", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Inputs dropped (nearest-first)")
    fig.text(0.02, 0.50, "Mean Dice (avg over 8 ROIs)", rotation="vertical",
             va="center", ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), borderaxespad=0)
    fig.tight_layout(rect=(0.08, 0.02, 1, 0.91), h_pad=0.8)
    out = save_pdf(fig, "exp1_dice_curves")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
