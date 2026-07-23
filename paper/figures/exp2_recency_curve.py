#!/usr/bin/env python3
"""Exp 2 -- Recency curve: EPE vs masked-input distance to target.

Run:
    python paper/figures/exp2_recency_curve.py
Output:
    paper/images/exp2_recency_curve.pdf
"""

import matplotlib.pyplot as plt

from _style import MODEL_COLOR, MODEL_LABEL, apply_style, load, save_pdf


def main():
    apply_style()
    per_main, _ = load("main",   "exp2")
    per_nu,   _ = load("nounet", "exp2")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for model, per, color in [
        ("main",   per_main, MODEL_COLOR["main"]),
        ("nounet", per_nu,   MODEL_COLOR["nounet"]),
    ]:
        sub = per[per.masked_tp != "none"].copy()
        agg = (sub.groupby("masked_distance_mo")["epe"]
                  .agg(["mean", "std", "count"]).reset_index())
        agg["sem"] = agg["std"] / agg["count"].pow(0.5)

        full = per[per.masked_tp == "none"]["epe"]
        full_mean = full.mean()
        full_sem  = full.std() / (len(full) ** 0.5)

        xs = [0] + agg.masked_distance_mo.tolist()
        ys = [full_mean] + agg["mean"].tolist()
        es = [full_sem]  + agg["sem"].tolist()

        ax.errorbar(xs, ys, yerr=es,
                    fmt="o-", color=color, capsize=2, lw=1.2, ms=5,
                    label=MODEL_LABEL[model])

    ax.set_xlabel("Masked-input distance from target (months)")
    ax.set_ylabel("EPE (voxels)")
    ax.set_title(r"Exp.~2 -- Recency effect: EPE vs.\ masked-input distance")
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = save_pdf(fig, "exp2_recency_curve")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
