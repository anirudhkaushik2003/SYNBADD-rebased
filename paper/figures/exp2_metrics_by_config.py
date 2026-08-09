#!/usr/bin/env python3
"""Exp 2 -- PSNR and EPE by single-input mask configuration, both models.

Run:
    python paper/figures/exp2_metrics_by_config.py
Output:
    paper/images/exp2_metrics_by_config.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from _style import GROUP_COLOR, MODEL_LABEL, apply_style, load, save_pdf


CONFIG_ORDER = ["none", "m06", "m18", "m24", "m36"]
CONFIG_NICE = {
    "none": "Full",
    "m06":  r"$-$m06" "\n" r"(past, 6mo)",
    "m18":  r"$-$m18" "\n" r"(future, 6mo)",
    "m24":  r"$-$m24" "\n" r"(future, 12mo)",
    "m36":  r"$-$m36" "\n" r"(future, 24mo)",
}


def main():
    apply_style()
    per_main, _ = load("main",   "exp2")
    per_nu,   _ = load("nounet", "exp2")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    metrics = [("psnr", "PSNR (dB)"), ("epe", "EPE (voxels)")]

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(CONFIG_ORDER))
        w = 0.20

        for model_index, (model, per) in enumerate(
            [("main", per_main), ("nounet", per_nu)]):
            agg = (per.groupby(["group", "masked_tp"])[metric]
                      .agg(["mean", "std"]).reset_index())
            for gi, grp in enumerate(["CN", "MCI"]):
                sub = agg[agg.group == grp]
                vals, stds = [], []
                for tp in CONFIG_ORDER:
                    rr = sub[sub.masked_tp == tp]
                    vals.append(rr["mean"].iloc[0] if not rr.empty else 0.0)
                    stds.append(rr["std"].iloc[0] if not rr.empty else 0)
                shift = (model_index * 2 + gi - 1.5) * w
                hatch = "" if model == "main" else "//"
                bars = ax.bar(
                    x + shift, vals, w * 0.85, yerr=stds, capsize=1.4,
                    color=GROUP_COLOR[grp], hatch=hatch, edgecolor="black",
                    lw=0.4, alpha=0.85,
                    label=f"{grp} ({MODEL_LABEL[model]})" if metric == "psnr" else None,
                )
                ax.bar_label(
                    bars, labels=[f"{value:.2f}" for value in vals],
                    padding=2, fontsize=7, rotation=90,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_NICE[c] for c in CONFIG_ORDER])
        ax.set_ylabel(label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    out = save_pdf(fig, "exp2_metrics_by_config")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
