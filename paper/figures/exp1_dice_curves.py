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

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
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
                label=f"{grp} ({MODEL_LABEL[model]})",
            )
        ax.set_xlabel("Inputs dropped (nearest-first)")
        ax.set_xticks(range(0, 5))

    axes[0].set_ylabel("Mean Dice (avg over 8 ROIs)")
    handles, labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out = save_pdf(fig, "exp1_dice_curves")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
