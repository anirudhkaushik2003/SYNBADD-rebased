#!/usr/bin/env python3
"""Exp 1 -- Per-region Dice grouped bar chart (replaces the radar plot).

For each diagnostic group (CN, MCI, AD), each ROI gets four bars:
    * UNet+DAFT, K=full
    * UNet+DAFT, K=1
    * No UNet+No DAFT, K=full
    * No UNet+No DAFT, K=1

Run:
    python paper/figures/exp1_per_region_bars.py
Output:
    paper/images/exp1_per_region_bars.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from _style import (MODEL_COLOR, MODEL_LABEL, REGIONS_ORDER, REGION_NICE,
                    apply_style, load, save_pdf)


def main():
    apply_style()
    _, dice_main = load("main", "exp1")
    _, dice_nu   = load("nounet", "exp1")

    groups = ["CN", "MCI", "AD"]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)

    region_labels = [REGION_NICE[r] for r in REGIONS_ORDER]
    n_reg = len(REGIONS_ORDER)
    bar_w = 0.20
    x = np.arange(n_reg)

    for ax, grp in zip(axes, groups):
        # Each model: K=full, K=1
        bars = []
        for offset, (model, dice, color) in enumerate([
            ("main",   dice_main, MODEL_COLOR["main"]),
            ("nounet", dice_nu,   MODEL_COLOR["nounet"]),
        ]):
            sub = dice[dice.group == grp]
            if sub.empty:
                continue
            full_n = 0
            min_n = int(sub.n_dropped.max())

            full_vals, full_sem = [], []
            min_vals,  min_sem  = [], []
            for r in REGIONS_ORDER:
                f = sub[(sub.n_dropped == full_n) & (sub.region == r)]["dice"]
                m = sub[(sub.n_dropped == min_n)  & (sub.region == r)]["dice"]
                full_vals.append(f.mean() if not f.empty else np.nan)
                full_sem.append(f.std() / np.sqrt(len(f)) if len(f) > 1 else 0)
                min_vals.append(m.mean() if not m.empty else np.nan)
                min_sem.append(m.std() / np.sqrt(len(m)) if len(m) > 1 else 0)

            shift_full = (offset * 2 + 0 - 1.5) * bar_w
            shift_min  = (offset * 2 + 1 - 1.5) * bar_w

            b1 = ax.bar(x + shift_full, full_vals, bar_w, yerr=full_sem,
                        capsize=1.6, color=color, edgecolor="black", lw=0.4,
                        alpha=0.95, label=f"{MODEL_LABEL[model]} (full)")
            b2 = ax.bar(x + shift_min,  min_vals,  bar_w, yerr=min_sem,
                        capsize=1.6, color=color, edgecolor="black", lw=0.4,
                        alpha=0.55, hatch="//", label=MODEL_LABEL[model] + r" ($K{=}1$)")
            bars.extend([b1, b2])

        ax.set_ylabel(f"{grp}\nDice")
        ax.set_ylim(0.78, 1.005)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(region_labels, rotation=30, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = save_pdf(fig, "exp1_per_region_bars")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
