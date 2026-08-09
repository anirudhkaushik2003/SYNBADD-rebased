#!/usr/bin/env python3
"""Exp 1 -- Image metrics (SSIM, PSNR, EPE) vs n_dropped, per group, both models.

Run:
    python paper/figures/exp1_image_metrics.py
Output:
    paper/images/exp1_image_metrics.pdf
"""

import matplotlib.pyplot as plt

from _style import GROUP_COLOR, MODEL_LABEL, apply_style, load, save_pdf


def main():
    apply_style()
    per_main, _ = load("main", "exp1")
    per_nu,   _ = load("nounet", "exp1")

    metrics = [
        ("ssim", "SSIM"),
        ("psnr", "PSNR (dB)"),
        ("epe",  "Endpoint Error (voxels)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 3.4))

    for ax, (m, label) in zip(axes, metrics):
        for model, per in [("main", per_main), ("nounet", per_nu)]:
            for grp in ["CN", "MCI", "AD"]:
                sub = per[per.group == grp]
                if sub.empty:
                    continue
                mean = sub.groupby("n_dropped")[m].mean().sort_index()
                std = sub.groupby("n_dropped")[m].std().sort_index().fillna(0)
                style = "-" if model == "main" else "--"
                ax.errorbar(
                    mean.index, mean.values, yerr=std.values,
                    fmt="o", linestyle=style, color=GROUP_COLOR[grp],
                    capsize=2, lw=1.0, ms=4, alpha=0.85,
                    label=f"{grp} ({MODEL_LABEL[model]})" if m == "ssim" else None,
                )
        ax.set_xlabel("Inputs dropped (nearest-first)")
        ax.set_ylabel(label)
        ax.set_xticks(range(0, 5))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = save_pdf(fig, "exp1_image_metrics")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
