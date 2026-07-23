#!/usr/bin/env python3
"""Exp 2 -- Scatter of EPE under past mask vs (mean) future mask.

Per-subject scatter for the MCI cohort under the No UNet+No DAFT model.
Tight clustering around y=x indicates no past-vs-future asymmetry.

Run:
    python paper/figures/exp2_past_vs_future_scatter.py
Output:
    paper/images/exp2_past_vs_future_scatter.pdf
"""

import matplotlib.pyplot as plt

from _style import MODEL_COLOR, apply_style, load, save_pdf


def main():
    apply_style()
    per_nu, _ = load("nounet", "exp2")

    sub = per_nu[per_nu.group == "MCI"].copy()
    pivot = sub.pivot_table(index="subject", columns="masked_tp",
                            values="epe", aggfunc="mean")

    if "m06" not in pivot.columns:
        print("no m06 column in MCI subset")
        return

    fut_cols = [c for c in ["m18", "m24", "m36"] if c in pivot.columns]
    past = pivot["m06"].dropna()
    fut = pivot[fut_cols].mean(axis=1).dropna()
    common = past.index.intersection(fut.index)
    past, fut = past.loc[common], fut.loc[common]

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(past, fut, color=MODEL_COLOR["nounet"], s=22,
               edgecolor="black", lw=0.4, alpha=0.85)

    lim = [min(past.min(), fut.min()) * 0.95, max(past.max(), fut.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.6, label=r"$y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"EPE under past-input mask ($-$m06)")
    ax.set_ylabel(r"EPE under future-input mask (mean over $-$m18, $-$m24, $-$m36)")
    ax.set_title("Exp.~2 -- Past vs.\\ future masking, MCI cohort\n(No UNet$+$No DAFT)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = save_pdf(fig, "exp2_past_vs_future_scatter")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
