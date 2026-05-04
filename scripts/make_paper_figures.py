#!/usr/bin/env python3
"""Generate all robustness-experiment figures for the SynBADD MedIA paper.

Reads per-config + per-config Dice CSVs from both models (Main UNet+DAFT
and the NoUnet+NoDAFT ablation) on fold 2, and writes vector PDF figures
to paper/images/.

Style: clean, consistent, journal-ready. No emojis. Vector PDF for crisp
typesetting; matplotlib does the line/bar plots; we use a polar projection
for the per-region radar plot.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path("/root/anirudhkaushik/SYNBADD-rebased")
IMG_DIR = REPO / "paper" / "images"
IMG_DIR.mkdir(exist_ok=True)

QUAL = Path("/scratch/anirudhkaushik/SYNBADD_outputs/SYNBADD_Qualitative_2")

EXP_DIRS = {
    ("main", "exp1"):
        QUAL / "experiment_reduced_time_input_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2",
    ("nounet", "exp1"):
        QUAL / "experiment_reduced_time_input_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2",
    ("main", "exp2"):
        QUAL / "experiment_asymmetric_information_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2",
    ("nounet", "exp2"):
        QUAL / "experiment_asymmetric_information_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2",
}

MODEL_LABEL = {
    "main": r"UNet$+$DAFT (full)",
    "nounet": r"No UNet$+$No DAFT",
}
MODEL_COLOR = {
    "main": "#0072B2",     # blue
    "nounet": "#D55E00",   # vermilion
}
GROUP_COLOR = {"CN": "#009E73", "MCI": "#E69F00", "AD": "#CC79A7"}

REGIONS_ORDER = [
    "hippocampus", "amygdala", "entorhinal_cortex", "parahippocampal",
    "fusiform_gyrus", "inferior_temporal_gyrus", "middle_temporal_gyrus",
    "third_ventricle",
]
REGION_NICE = {
    "hippocampus": "Hippocampus",
    "amygdala": "Amygdala",
    "entorhinal_cortex": "Entorhinal\nCortex",
    "parahippocampal": "Parahippocampal",
    "fusiform_gyrus": "Fusiform\nGyrus",
    "inferior_temporal_gyrus": "Inferior\nTemporal G.",
    "middle_temporal_gyrus": "Middle\nTemporal G.",
    "third_ventricle": "Third\nVentricle",
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed TrueType -> editable
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load(model, exp):
    d = EXP_DIRS[(model, exp)]
    per = pd.read_csv(d / "per_config_metrics.csv")
    dice_path = d / "per_config_dice_volume.csv"
    dice = pd.read_csv(dice_path) if dice_path.exists() else None
    return per, dice


def save(fig, name):
    out = IMG_DIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 1: Exp1 — image metrics vs n_dropped, per group, both models
# ---------------------------------------------------------------------------

def fig_exp1_image_metrics():
    per_main, _ = load("main", "exp1")
    per_nu, _ = load("nounet", "exp1")

    metrics = [("ssim", "SSIM"), ("psnr", "PSNR (dB)"), ("epe", "Endpoint Error (voxels)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 3.4), sharex=False)

    for ax, (m, label) in zip(axes, metrics):
        for model, per in [("main", per_main), ("nounet", per_nu)]:
            for grp in ["CN", "MCI", "AD"]:
                sub = per[per.group == grp]
                if sub.empty:
                    continue
                grp_mean = sub.groupby("n_dropped")[m].mean().sort_index()
                grp_std = sub.groupby("n_dropped")[m].std().sort_index().fillna(0)
                style = "-" if model == "main" else "--"
                ax.errorbar(
                    grp_mean.index, grp_mean.values, yerr=grp_std.values,
                    fmt="o", linestyle=style, color=GROUP_COLOR[grp],
                    capsize=2, lw=1.0, ms=4,
                    label=f"{grp} ({MODEL_LABEL[model]})" if m == "ssim" else None,
                    alpha=0.85,
                )
        ax.set_xlabel("Inputs dropped (nearest-first)")
        ax.set_ylabel(label)
        ax.set_xticks(range(0, 5))
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Exp.~1 -- Image-level metrics under reduced input timepoints", y=1.02)
    fig.tight_layout()
    save(fig, "exp1_image_metrics")


# ---------------------------------------------------------------------------
# Figure 2: Exp1 — Dice (avg across ROIs) vs n_dropped, per group, both models
# ---------------------------------------------------------------------------

def fig_exp1_dice_curves():
    _, dice_main = load("main", "exp1")
    _, dice_nu = load("nounet", "exp1")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    titles = [MODEL_LABEL["main"], MODEL_LABEL["nounet"]]
    for ax, dice, ttl in zip(axes, [dice_main, dice_nu], titles):
        avg = (dice.groupby(["subject", "group", "n_dropped"])["dice"].mean()
                  .reset_index())
        for grp in ["CN", "MCI", "AD"]:
            sub = avg[avg.group == grp]
            if sub.empty:
                continue
            mean = sub.groupby("n_dropped")["dice"].mean().sort_index()
            sem = (sub.groupby("n_dropped")["dice"].std()
                      / np.sqrt(sub.groupby("n_dropped")["dice"].count())).sort_index()
            ax.errorbar(mean.index, mean.values, yerr=sem.values,
                        fmt="o-", color=GROUP_COLOR[grp], capsize=2, lw=1.0, ms=4,
                        label=grp)
        ax.set_xlabel("Inputs dropped (nearest-first)")
        ax.set_title(ttl)
        ax.set_xticks(range(0, 5))
    axes[0].set_ylabel("Mean Dice (avg over 8 ROIs)")
    axes[1].legend(title="Group", loc="lower left")
    fig.suptitle("Exp.~1 -- Dice degradation under reduced input timepoints", y=1.02)
    fig.tight_layout()
    save(fig, "exp1_dice_curves")


# ---------------------------------------------------------------------------
# Figure 3: Per-region radar plot — Dice at K=full vs K=1 (NoUnet+NoDAFT, MCI)
# Best illustration of where the model degrades.
# ---------------------------------------------------------------------------

def _radar_axes(ax, labels):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(90)
    return angles, angles_closed


def fig_radar_per_region():
    _, dice_main = load("main", "exp1")
    _, dice_nu = load("nounet", "exp1")

    fig = plt.figure(figsize=(11.5, 4.0))
    gs = fig.add_gridspec(1, 3, wspace=0.55)
    titles = [
        ("CN", "Cognitively Normal"),
        ("MCI", "Mild Cognitive Impairment"),
        ("AD", "Alzheimer's Disease"),
    ]

    region_labels = [REGION_NICE[r] for r in REGIONS_ORDER]

    for col, (grp, full_label) in enumerate(titles):
        ax = fig.add_subplot(gs[0, col], projection="polar")
        angles, angles_closed = _radar_axes(ax, region_labels)
        ax.set_ylim(0.78, 1.0)
        ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"], fontsize=7)

        for model, dice, color in [
            ("main", dice_main, MODEL_COLOR["main"]),
            ("nounet", dice_nu, MODEL_COLOR["nounet"]),
        ]:
            sub = dice[dice.group == grp]
            if sub.empty:
                continue
            full_n = 0
            min_n = int(sub.n_dropped.max())
            for n_drop, line, fillalpha in [(full_n, "-", 0.08), (min_n, ":", 0.04)]:
                vals = []
                for r in REGIONS_ORDER:
                    rs = sub[(sub.n_dropped == n_drop) & (sub.region == r)]
                    vals.append(rs.dice.mean() if not rs.empty else np.nan)
                vals_closed = vals + [vals[0]]
                if n_drop == 0:
                    tag = f"{MODEL_LABEL[model]} (full)"
                else:
                    tag = MODEL_LABEL[model] + r" ($K\!=\!1$)"
                ax.plot(angles_closed, vals_closed, line, color=color, lw=1.4, label=tag)
                ax.fill(angles_closed, vals_closed, color=color, alpha=fillalpha)

        ax.set_title(f"{full_label} ({grp})", pad=18, fontsize=10)
        ax.tick_params(axis="x", pad=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Exp.~1 -- Per-region Dice at full context vs.\\ source-only ($K{=}1$)", y=1.05)
    save(fig, "exp1_radar_per_region")


# ---------------------------------------------------------------------------
# Figure 4: Cross-model paired Δ — bar chart with paired-Wilcoxon p-values
# ---------------------------------------------------------------------------

def fig_cross_model_delta():
    sig_csv = Path("/scratch/anirudhkaushik/SYNBADD_outputs/significance_tests_fold2/significance_tests.csv")
    df = pd.read_csv(sig_csv)
    cm = df[(df.test == "cross_model_delta_wilcoxon") &
            (df.experiment == "reduced_time_input")]
    pooled = cm[cm.scope == "all_cohort_stable_n0_n2"].copy()

    metrics = [("ssim", "SSIM"), ("psnr", "PSNR"), ("epe", "EPE"),
               ("dice_avg_roi", "Dice (avg ROIs)")]
    pooled["m_label"] = pooled["metric"].map(dict(metrics))
    pooled = pooled.dropna(subset=["m_label"]).set_index("m_label").reindex([m[1] for m in metrics])

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(pooled))
    w = 0.36

    main_d = pooled["model_a_mean_delta"].values
    nu_d = pooled["model_b_mean_delta"].values

    main_disp = main_d.copy()
    nu_disp = nu_d.copy()
    for i, m in enumerate(["SSIM", "PSNR", "EPE", "Dice (avg ROIs)"]):
        if m == "PSNR":
            pass  # already in dB
        elif m in {"EPE"}:
            pass

    bars1 = ax.bar(x - w/2, main_disp, w, color=MODEL_COLOR["main"],
                   label=MODEL_LABEL["main"], edgecolor="black", lw=0.4)
    bars2 = ax.bar(x + w/2, nu_disp, w, color=MODEL_COLOR["nounet"],
                   label=MODEL_LABEL["nounet"], edgecolor="black", lw=0.4)

    # annotate p-values above each pair
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        p = pooled["p_value"].iloc[i]
        if pd.isna(p):
            continue
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        top = max(abs(b1.get_height()), abs(b2.get_height()))
        ax.text(i, top * 1.18 if top > 0 else 0.001, f"$p${marker}",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel(r"Mean per-subject $\Delta$ ($n_{\mathrm{drop}}{=}0 \rightarrow 2$)")
    ax.set_title("Exp.~1 -- Cross-model paired-$\\Delta$ (cohort-stable, $n{=}76$)")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    save(fig, "exp1_cross_model_delta")


# ---------------------------------------------------------------------------
# Figure 5: Exp 2 — image metrics by mask configuration, per group, both models
# ---------------------------------------------------------------------------

def fig_exp2_metrics_by_config():
    per_main, _ = load("main", "exp2")
    per_nu, _ = load("nounet", "exp2")

    order = ["none", "m06", "m18", "m24", "m36"]
    nice = {"none": "Full", "m06": "$-$m06\n(past, 6mo)",
            "m18": "$-$m18\n(future, 6mo)", "m24": "$-$m24\n(future, 12mo)",
            "m36": "$-$m36\n(future, 24mo)"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=False)
    metrics = [("psnr", "PSNR (dB)"), ("epe", "Endpoint Error (voxels)")]

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(order))
        w = 0.28

        for offset, (model, per, color) in enumerate([
            ("main", per_main, MODEL_COLOR["main"]),
            ("nounet", per_nu, MODEL_COLOR["nounet"]),
        ]):
            agg = (per.groupby(["group", "masked_tp"])[metric]
                      .agg(["mean", "std"]).reset_index())
            for gi, grp in enumerate(["CN", "MCI"]):
                sub = agg[agg.group == grp]
                vals = []
                stds = []
                for tp in order:
                    rr = sub[sub.masked_tp == tp]
                    vals.append(rr["mean"].iloc[0] if not rr.empty else np.nan)
                    stds.append(rr["std"].iloc[0] if not rr.empty else 0)
                shift = (offset * 2 + gi - 1.5) * w
                hatch = "" if model == "main" else "//"
                edge = "black"
                ax.bar(x + shift, vals, w * 0.85,
                       yerr=stds, capsize=1.4, color=color,
                       hatch=hatch, edgecolor=edge, lw=0.4, alpha=0.85,
                       label=f"{grp} ({MODEL_LABEL[model]})" if metric == "psnr" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([nice[c] for c in order])
        ax.set_ylabel(label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Exp.~2 -- Image metrics under single-input masking", y=1.02)
    fig.tight_layout()
    save(fig, "exp2_metrics_by_config")


# ---------------------------------------------------------------------------
# Figure 6: Exp 2 — recency curve (NoUnet+NoDAFT) — EPE vs masked distance
# ---------------------------------------------------------------------------

def fig_exp2_recency_curve():
    per_main, _ = load("main", "exp2")
    per_nu, _ = load("nounet", "exp2")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for model, per, color in [
        ("main", per_main, MODEL_COLOR["main"]),
        ("nounet", per_nu, MODEL_COLOR["nounet"]),
    ]:
        sub = per[per.masked_tp != "none"].copy()
        # absolute distance is masked_distance_mo
        agg = sub.groupby("masked_distance_mo")["epe"].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        # full = none (distance 0)
        full_row = per[per.masked_tp == "none"]["epe"].agg(["mean", "std", "count"])
        full_sem = full_row["std"] / np.sqrt(full_row["count"])
        x_full = [0]
        y_full = [full_row["mean"]]
        y_full_err = [full_sem]
        ax.errorbar(x_full + agg.masked_distance_mo.tolist(),
                    y_full + agg["mean"].tolist(),
                    yerr=y_full_err + agg["sem"].tolist(),
                    fmt="o-", color=color, capsize=2, lw=1.2, ms=5,
                    label=MODEL_LABEL[model])

    ax.set_xlabel("Masked-input distance from target (months)")
    ax.set_ylabel("EPE (voxels)")
    ax.set_title("Exp.~2 -- Recency effect: EPE vs.\\ masked-input distance")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "exp2_recency_curve")


# ---------------------------------------------------------------------------
# Figure 7: Exp 2 — past vs future scatter (NoUnet+NoDAFT MCI)
# ---------------------------------------------------------------------------

def fig_exp2_past_vs_future():
    per_nu, _ = load("nounet", "exp2")
    sub = per_nu[per_nu.group == "MCI"].copy()
    pivot = sub.pivot_table(index="subject", columns="masked_tp", values="epe", aggfunc="mean")
    if "m06" not in pivot.columns:
        return
    fut_cols = [c for c in ["m18", "m24", "m36"] if c in pivot.columns]
    past = pivot["m06"]
    fut = pivot[fut_cols].mean(axis=1)
    common = past.dropna().index.intersection(fut.dropna().index)
    past, fut = past.loc[common], fut.loc[common]

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(past, fut, color=MODEL_COLOR["nounet"], s=22, edgecolor="black", lw=0.4, alpha=0.85)
    lim = [min(past.min(), fut.min()) * 0.95, max(past.max(), fut.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.6, label=r"$y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("EPE under past-input mask ($-$m06)")
    ax.set_ylabel("EPE under future-input mask (mean over $-$m18, $-$m24, $-$m36)")
    ax.set_title("Exp.~2 -- Past vs.\\ future masking, MCI cohort\n(No UNet$+$No DAFT)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "exp2_past_vs_future_scatter")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    fig_exp1_image_metrics()
    fig_exp1_dice_curves()
    fig_radar_per_region()
    fig_cross_model_delta()
    fig_exp2_metrics_by_config()
    fig_exp2_recency_curve()
    fig_exp2_past_vs_future()


if __name__ == "__main__":
    main()
