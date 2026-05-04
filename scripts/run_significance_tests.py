#!/usr/bin/env python3
"""Statistical-significance + correlation analysis for the robustness experiments.

Pulls per-config metrics for every (model, experiment) combination we have on
disk, runs the relevant within-subject tests, cross-model paired comparisons,
and asymmetry tests. Writes:

    - CSV   : long-format table of every test result
    - MD    : human-readable summary report (paper-ready)
    - W&B   : tables + scalars

Reruns are safe; the script will skip cells whose CSVs are missing.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT_BASE = "/scratch/anirudhkaushik/SYNBADD_outputs/SYNBADD_Qualitative_2"

# (model_label, experiment_name, run_name)
RUNS = [
    ("main",          "reduced_time_input",
     "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2"),
    ("main",          "asymmetric_information",
     "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2"),
    ("nounet_nodaft", "reduced_time_input",
     "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2"),
    ("nounet_nodaft", "asymmetric_information",
     "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2"),
]

DEFAULT_OUTPUT_DIR = "/scratch/anirudhkaushik/SYNBADD_outputs/significance_tests_fold2"

ALPHA = 0.05
IMG_METRICS = ["ssim", "psnr", "mse", "nrmse", "epe"]
DICE_METRICS = ["dice", "vol_diff_pct"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data():
    """Return {(model, exp): {'per': df, 'dice': df}} for all available cells."""
    data = {}
    for model, exp, run in RUNS:
        d = f"{OUTPUT_BASE}/experiment_{exp}_fold_2/{run}"
        per_path = os.path.join(d, "per_config_metrics.csv")
        if not os.path.exists(per_path):
            print(f"  [skip] {model}/{exp}: per_config_metrics.csv missing")
            continue
        per = pd.read_csv(per_path)
        dice_path = os.path.join(d, "per_config_dice_volume.csv")
        dice = pd.read_csv(dice_path) if os.path.exists(dice_path) else None
        data[(model, exp)] = {"per": per, "dice": dice, "dir": d}
        print(f"  [load] {model}/{exp}: per={len(per)} rows, dice={'-' if dice is None else len(dice)} rows")
    return data


def agg_key_for(experiment):
    return "n_dropped" if experiment == "reduced_time_input" else "masked_tp"


def is_numeric_axis(experiment):
    return experiment == "reduced_time_input"


# ---------------------------------------------------------------------------
# Test primitives
# ---------------------------------------------------------------------------

def friedman_within(df, agg_key, metric):
    """Repeated-measures non-parametric test across configs.

    Each subject is a block; configs are within-block treatments.
    Restricts to subjects with values for every config (balanced).
    """
    if metric not in df.columns:
        return None
    pivot = df.pivot_table(index="subject", columns=agg_key, values=metric, aggfunc="mean")
    pivot = pivot.dropna()
    if pivot.shape[1] < 2 or pivot.shape[0] < 3:
        return None
    cols = list(pivot.columns)
    try:
        stat, p = stats.friedmanchisquare(*[pivot[c].values for c in cols])
    except ValueError:
        # All-identical inputs raise; treat as flat.
        return {"statistic": 0.0, "p_value": 1.0,
                "n_subjects": int(pivot.shape[0]), "n_configs": int(pivot.shape[1]),
                "max_minus_min": 0.0}
    if (pivot.values - pivot.values[:, :1] == 0).all():
        # Identical across configs -> p is undefined; report degenerate.
        return {"statistic": 0.0, "p_value": 1.0,
                "n_subjects": int(pivot.shape[0]), "n_configs": int(pivot.shape[1]),
                "max_minus_min": 0.0}
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_subjects": int(pivot.shape[0]),
        "n_configs": int(pivot.shape[1]),
        "max_minus_min": float(pivot.mean().max() - pivot.mean().min()),
    }


def wilcoxon_paired(df, agg_key, metric, val_a, val_b):
    """Paired Wilcoxon between two configs for the same metric."""
    if metric not in df.columns:
        return None
    pivot = df.pivot_table(index="subject", columns=agg_key, values=metric, aggfunc="mean")
    if val_a not in pivot.columns or val_b not in pivot.columns:
        return None
    paired = pivot[[val_a, val_b]].dropna()
    if len(paired) < 5:
        return None
    diff = paired[val_a].values - paired[val_b].values
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": len(paired),
                "median_diff": 0.0, "mean_diff": 0.0,
                "fraction_a_gt_b": float((diff > 0).mean())}
    try:
        stat, p = stats.wilcoxon(paired[val_a], paired[val_b])
    except ValueError:
        return None
    return {
        "statistic": float(stat), "p_value": float(p),
        "n": int(len(paired)),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(diff.mean()),
        "fraction_a_gt_b": float((diff > 0).mean()),
    }


def per_subject_spearman(df, x_col, y_col):
    """Per-subject Spearman ρ between x_col and y_col, then 1-sample t-test against 0."""
    if x_col not in df.columns or y_col not in df.columns:
        return None
    rhos = []
    for _, sub in df.groupby("subject"):
        x = pd.to_numeric(sub[x_col], errors="coerce")
        y = pd.to_numeric(sub[y_col], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(x[m], y[m])
        if not np.isnan(rho):
            rhos.append(rho)
    if len(rhos) < 3:
        return None
    rhos = np.asarray(rhos)
    if np.allclose(rhos, 0):
        return {"n_subjects": int(len(rhos)), "mean_rho": 0.0, "median_rho": 0.0,
                "t_stat": 0.0, "p_value": 1.0,
                "frac_negative_rho": 0.0, "frac_positive_rho": 0.0}
    t_stat, t_p = stats.ttest_1samp(rhos, 0)
    return {
        "n_subjects": int(len(rhos)),
        "mean_rho": float(rhos.mean()),
        "median_rho": float(np.median(rhos)),
        "t_stat": float(t_stat),
        "p_value": float(t_p),
        "frac_negative_rho": float((rhos < 0).mean()),
        "frac_positive_rho": float((rhos > 0).mean()),
    }


def cross_model_delta_test(per_a, per_b, agg_key, val_max, val_min, metric):
    """Compare per-subject Δ(val_max → val_min) between two models on the same subjects."""
    def _get_delta(df):
        if metric not in df.columns:
            return None
        p = df.pivot_table(index="subject", columns=agg_key, values=metric, aggfunc="mean")
        if val_max not in p.columns or val_min not in p.columns:
            return None
        return (p[val_max] - p[val_min]).dropna()

    a = _get_delta(per_a)
    b = _get_delta(per_b)
    if a is None or b is None:
        return None
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    a, b = a.loc[common], b.loc[common]
    if np.allclose(a.values - b.values, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(common)),
                "model_a_mean_delta": float(a.mean()),
                "model_b_mean_delta": float(b.mean()),
                "delta_of_deltas_mean": 0.0}
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:
        return None
    return {
        "statistic": float(stat), "p_value": float(p),
        "n": int(len(common)),
        "model_a_mean_delta": float(a.mean()),
        "model_b_mean_delta": float(b.mean()),
        "delta_of_deltas_mean": float((b - a).mean()),
    }


# ---------------------------------------------------------------------------
# Test orchestration
# ---------------------------------------------------------------------------

def collect_tests(data):
    """Return a long-format DataFrame: one row per test."""
    rows = []

    def add(model, exp, test_name, scope, metric, hypothesis, result, comment=""):
        if result is None:
            return
        sig = "***" if result.get("p_value", 1) < 0.001 else \
              "**"  if result.get("p_value", 1) < 0.01  else \
              "*"   if result.get("p_value", 1) < ALPHA else ""
        rows.append({
            "model": model, "experiment": exp,
            "test": test_name, "scope": scope, "metric": metric,
            "hypothesis": hypothesis,
            **result,
            "significant": bool(result.get("p_value", 1) < ALPHA),
            "sig_marker": sig,
            "comment": comment,
        })

    # ---------- within-model tests ----------
    for (model, exp), payload in data.items():
        per = payload["per"]
        dice = payload["dice"]
        agg_key = agg_key_for(exp)

        # Per-subject avg-Dice across ROIs (one value per (subject, config))
        if dice is not None and not dice.empty:
            mean_dice_per_cfg = (
                dice.groupby(["subject", "group", "age_bin", "config_id", agg_key])["dice"]
                .mean().reset_index()
            )
        else:
            mean_dice_per_cfg = None

        groups = list(per.group.unique())

        # ---- Friedman: does metric vary across configs? ----
        for metric in IMG_METRICS:
            for scope, sub in [("all", per)] + [(f"group:{g}", per[per.group == g]) for g in groups]:
                r = friedman_within(sub, agg_key, metric)
                add(model, exp, "friedman_within_subject", scope, metric,
                    f"{metric} differs across {agg_key}",
                    r,
                    comment="Repeated-measures non-parametric test; flat null = invariance")

        if mean_dice_per_cfg is not None:
            for scope, sub in [("all", mean_dice_per_cfg)] + \
                              [(f"group:{g}", mean_dice_per_cfg[mean_dice_per_cfg.group == g]) for g in groups]:
                r = friedman_within(sub, agg_key, "dice")
                add(model, exp, "friedman_within_subject", scope, "dice_avg_roi",
                    f"avg-ROI Dice differs across {agg_key}",
                    r,
                    comment="Per-subject avg over 8 ROIs")

        # ---- Per-region Friedman on Dice ----
        if dice is not None:
            for region in dice.region.unique():
                sub_r = dice[dice.region == region]
                r = friedman_within(sub_r, agg_key, "dice")
                add(model, exp, "friedman_within_subject", f"region:{region}", "dice",
                    f"region={region} Dice differs across {agg_key}",
                    r)

        # ---- Wilcoxon paired: extreme-config contrasts ----
        if exp == "reduced_time_input":
            # Per-group K=max vs K=1 (different K_max per group)
            for grp in groups:
                sub = per[per.group == grp]
                if sub.empty:
                    continue
                vmax, vmin = sub[agg_key].min(), sub[agg_key].max()  # n_dropped=0 vs max
                for metric in IMG_METRICS:
                    r = wilcoxon_paired(sub, agg_key, metric, vmax, vmin)
                    add(model, exp, "wilcoxon_paired", f"group:{grp}", metric,
                        f"{metric}(n=0) vs {metric}(n={vmin if False else vmax}={int(vmax)}) [n_drop=0 vs max]",
                        r)
                if mean_dice_per_cfg is not None:
                    sub_d = mean_dice_per_cfg[mean_dice_per_cfg.group == grp]
                    r = wilcoxon_paired(sub_d, agg_key, "dice", vmax, vmin)
                    add(model, exp, "wilcoxon_paired", f"group:{grp}", "dice_avg_roi",
                        f"avg-ROI Dice (full vs K=1) for {grp}",
                        r)
        else:
            # Exp 2: full vs each masked config
            for masked in ["m06", "m18", "m24", "m36"]:
                for scope, sub in [("all", per)] + [(f"group:{g}", per[per.group == g]) for g in groups]:
                    if sub.empty or masked not in set(sub[agg_key].unique()):
                        continue
                    for metric in IMG_METRICS:
                        r = wilcoxon_paired(sub, agg_key, metric, "none", masked)
                        add(model, exp, "wilcoxon_paired", scope, metric,
                            f"{metric}: full vs drop_{masked}",
                            r)
                    if mean_dice_per_cfg is not None:
                        sub_d = mean_dice_per_cfg[mean_dice_per_cfg.group == sub.group.iloc[0]] \
                            if scope.startswith("group:") else mean_dice_per_cfg
                        r = wilcoxon_paired(sub_d, agg_key, "dice", "none", masked)
                        add(model, exp, "wilcoxon_paired", scope, "dice_avg_roi",
                            f"avg-ROI Dice: full vs drop_{masked}",
                            r)

        # ---- Spearman ρ: per-subject correlation of x with metric ----
        if exp == "reduced_time_input":
            for metric in IMG_METRICS:
                for scope, sub in [("all", per)] + [(f"group:{g}", per[per.group == g]) for g in groups]:
                    r = per_subject_spearman(sub, "n_dropped", metric)
                    add(model, exp, "spearman_per_subject", scope, metric,
                        f"per-subject ρ(n_dropped, {metric})",
                        r,
                        comment="Negative ρ = degradation with more dropped")
            if mean_dice_per_cfg is not None:
                for scope, sub in [("all", mean_dice_per_cfg)] + \
                                  [(f"group:{g}", mean_dice_per_cfg[mean_dice_per_cfg.group == g]) for g in groups]:
                    r = per_subject_spearman(sub, "n_dropped", "dice")
                    add(model, exp, "spearman_per_subject", scope, "dice_avg_roi",
                        "per-subject ρ(n_dropped, avg-ROI Dice)",
                        r)
        else:
            # Distance-from-target correlation (Exp 2)
            df_d = per[per[agg_key] != "none"].copy()
            for metric in IMG_METRICS:
                for scope, sub in [("all", df_d)] + [(f"group:{g}", df_d[df_d.group == g]) for g in groups]:
                    r = per_subject_spearman(sub, "masked_distance_mo", metric)
                    add(model, exp, "spearman_per_subject", scope, metric,
                        f"per-subject ρ(masked_distance_mo, {metric})",
                        r,
                        comment="Tests near-vs-far asymmetry")
            if mean_dice_per_cfg is not None:
                # join distance via masked_tp lookup
                tp_to_dist = {"m06": 6, "m12": 0, "m18": 6, "m24": 12, "m36": 24}
                tp_to_dir = {"m06": "past", "m12": "none", "m18": "future",
                             "m24": "future", "m36": "future", "none": "none"}
                tmp = mean_dice_per_cfg.rename(columns={agg_key: "masked_tp"}).copy()
                tmp = tmp[tmp["masked_tp"] != "none"].copy()
                tmp["masked_distance_mo"] = tmp["masked_tp"].map(tp_to_dist)
                for scope, sub in [("all", tmp)] + [(f"group:{g}", tmp[tmp.group == g]) for g in groups]:
                    r = per_subject_spearman(sub, "masked_distance_mo", "dice")
                    add(model, exp, "spearman_per_subject", scope, "dice_avg_roi",
                        "per-subject ρ(masked_distance_mo, avg-ROI Dice)",
                        r)

        # ---- Past vs Future contrast (Exp 2 only) ----
        if exp == "asymmetric_information":
            for scope, sub in [("all", per)] + [(f"group:{g}", per[per.group == g]) for g in groups]:
                if sub.empty:
                    continue
                pivot = sub.pivot_table(index="subject", columns="masked_tp",
                                        values="ssim", aggfunc="mean")
                past_cols = [c for c in ["m06"] if c in pivot.columns]
                fut_cols = [c for c in ["m18", "m24", "m36"] if c in pivot.columns]
                if not past_cols or not fut_cols:
                    continue
                past = pivot[past_cols].mean(axis=1).dropna()
                fut = pivot[fut_cols].mean(axis=1).dropna()
                common = past.index.intersection(fut.index)
                if len(common) < 5:
                    continue
                past, fut = past.loc[common], fut.loc[common]
                if np.allclose(past.values - fut.values, 0):
                    add(model, exp, "wilcoxon_paired", scope, "ssim_past_vs_future",
                        "SSIM under past-mask vs future-mask",
                        {"statistic": 0.0, "p_value": 1.0, "n": int(len(common)),
                         "median_diff": 0.0, "mean_diff": 0.0,
                         "fraction_a_gt_b": float((past.values - fut.values > 0).mean())},
                        comment="No asymmetry detected (identical means)")
                    continue
                stat, p = stats.wilcoxon(past, fut)
                add(model, exp, "wilcoxon_paired", scope, "ssim_past_vs_future",
                    "SSIM under past-mask vs future-mask",
                    {"statistic": float(stat), "p_value": float(p),
                     "n": int(len(common)),
                     "median_diff": float(np.median(past.values - fut.values)),
                     "mean_diff": float((past - fut).mean()),
                     "fraction_a_gt_b": float((past.values > fut.values).mean())},
                    comment="Past-mask higher = future inputs more important")

    # ---------- Cross-model tests (same subjects across models) ----------
    cross_pairs = [("main", "nounet_nodaft")]
    for exp in {e for (_, e) in data.keys()}:
        for ma, mb in cross_pairs:
            if (ma, exp) not in data or (mb, exp) not in data:
                continue
            per_a = data[(ma, exp)]["per"]
            per_b = data[(mb, exp)]["per"]
            agg_key = agg_key_for(exp)
            groups = sorted(set(per_a.group.unique()) | set(per_b.group.unique()))

            if exp == "reduced_time_input":
                # Per-group max vs min using each group's own range (max(n_dropped) varies).
                for grp in groups:
                    sub_a = per_a[per_a.group == grp]
                    sub_b = per_b[per_b.group == grp]
                    if sub_a.empty or sub_b.empty:
                        continue
                    vmin = 0
                    vmax = int(min(sub_a[agg_key].max(), sub_b[agg_key].max()))
                    for metric in IMG_METRICS:
                        r = cross_model_delta_test(sub_a, sub_b, agg_key, vmin, vmax, metric)
                        add(f"{ma}_vs_{mb}", exp, "cross_model_delta_wilcoxon",
                            f"group:{grp}", metric,
                            f"is Δ{metric}(K=max→K=min) different between models?",
                            r,
                            comment=f"Δ_{ma} vs Δ_{mb} per subject; n_dropped {vmin}->{vmax}")

                # Pooled cohort-stable (n=0 vs n=2; all groups present)
                for metric in IMG_METRICS:
                    r = cross_model_delta_test(per_a, per_b, agg_key, 0, 2, metric)
                    add(f"{ma}_vs_{mb}", exp, "cross_model_delta_wilcoxon",
                        "all_cohort_stable_n0_n2", metric,
                        f"is Δ{metric}(n=0→n=2) different between models? (cohort-stable)",
                        r,
                        comment="76-subject pooled, equal cohort across both ends")

                # Same comparison on avg-ROI Dice
                if data[(ma, exp)]["dice"] is not None and data[(mb, exp)]["dice"] is not None:
                    md_a = (data[(ma, exp)]["dice"]
                            .groupby(["subject", "group", agg_key])["dice"].mean().reset_index())
                    md_b = (data[(mb, exp)]["dice"]
                            .groupby(["subject", "group", agg_key])["dice"].mean().reset_index())
                    r = cross_model_delta_test(md_a, md_b, agg_key, 0, 2, "dice")
                    add(f"{ma}_vs_{mb}", exp, "cross_model_delta_wilcoxon",
                        "all_cohort_stable_n0_n2", "dice_avg_roi",
                        "is Δ(avg-ROI Dice)(n=0→n=2) different between models?",
                        r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown(results, out_path):
    md = ["# SynBADD robustness experiments — statistical-test report",
          "", "Fold 2. α = 0.05 throughout. Sig markers: `*` p<0.05, `**` p<0.01, `***` p<0.001.",
          ""]

    cells = sorted(results[["model", "experiment"]].drop_duplicates().itertuples(index=False))
    md.append(f"**Cells analysed:** {len(cells)}")
    for c in cells:
        md.append(f"- `{c.model}` × `{c.experiment}`")
    md.append("")

    # Section: Friedman headlines
    md.append("## Within-model invariance / variation tests (Friedman, repeated measures)")
    md.append("")
    md.append("Tests whether per-subject metric values vary across configurations. "
              "*Significant = the model is responsive to the configuration axis. "
              "Non-significant or 0 = invariant (null robustness).*")
    md.append("")
    fried = results[results.test == "friedman_within_subject"]
    md.append("### Headline (scope=all)")
    md.append("")
    head = fried[fried.scope == "all"].copy()
    if not head.empty:
        head_disp = head[["model","experiment","metric","statistic","p_value","sig_marker","max_minus_min","n_subjects","n_configs"]]
        md.append(head_disp.to_markdown(index=False, floatfmt=".4g"))
        md.append("")

    # Per-group summary
    md.append("### Per-group breakdown")
    md.append("")
    sub = fried[fried.scope.str.startswith("group:")].copy()
    if not sub.empty:
        sub["group"] = sub.scope.str.replace("group:", "", regex=False)
        sub_disp = sub[["model","experiment","group","metric","p_value","sig_marker","max_minus_min","n_subjects"]]
        md.append(sub_disp.to_markdown(index=False, floatfmt=".4g"))
        md.append("")

    # Section: Spearman per-subject correlations
    md.append("## Per-subject Spearman correlations (then aggregated by 1-sample t-test on ρ vs 0)")
    md.append("")
    md.append("For Exp 1: ρ between `n_dropped` and metric — negative = robust degradation. "
              "For Exp 2: ρ between `masked_distance_mo` and metric — non-zero = near/far asymmetry.")
    md.append("")
    sp = results[results.test == "spearman_per_subject"]
    if not sp.empty:
        sp_disp = sp[["model","experiment","scope","metric","mean_rho","median_rho","p_value","sig_marker","n_subjects","frac_negative_rho"]]
        md.append(sp_disp.to_markdown(index=False, floatfmt=".4g"))
        md.append("")

    # Section: Pairwise contrasts (Wilcoxon)
    md.append("## Pairwise Wilcoxon contrasts")
    md.append("")
    md.append("Reduced-input experiment: `K=full vs K=1` per group. "
              "Asymmetric experiment: `full vs drop_<tp>` and `past vs future` SSIM.")
    md.append("")
    wx = results[results.test == "wilcoxon_paired"]
    if not wx.empty:
        wx_disp = wx[["model","experiment","scope","metric","hypothesis","p_value","sig_marker","mean_diff","n"]]
        md.append(wx_disp.to_markdown(index=False, floatfmt=".4g"))
        md.append("")

    # Section: Cross-model
    md.append("## Cross-model paired-Δ tests (same subjects)")
    md.append("")
    md.append("Per subject, compute Δmetric(K=max→K=min) under each model. "
              "Wilcoxon paired: are the two Δs different? Significant = the models *respond* to "
              "input dropping in distinguishable ways.")
    md.append("")
    cm = results[results.test == "cross_model_delta_wilcoxon"]
    if not cm.empty:
        cm_disp = cm[["model","experiment","scope","metric","hypothesis","model_a_mean_delta","model_b_mean_delta","p_value","sig_marker","n"]]
        md.append(cm_disp.to_markdown(index=False, floatfmt=".4g"))
        md.append("")

    # Auto-generated takeaways
    md.append("## Auto-generated takeaways")
    md.append("")
    md += _summary_bullets(results)

    Path(out_path).write_text("\n".join(md))


def _summary_bullets(results):
    bullets = []
    fried = results[results.test == "friedman_within_subject"]

    # 1) per-model invariance summary
    for (model, exp), sub in fried.groupby(["model", "experiment"]):
        all_scope = sub[sub.scope == "all"]
        if all_scope.empty:
            continue
        sig_metrics = all_scope[all_scope.significant]["metric"].tolist()
        flat_metrics = all_scope[~all_scope.significant]["metric"].tolist()
        if sig_metrics:
            bullets.append(
                f"- **{model} / {exp}**: configurations significantly affect "
                f"{', '.join(sig_metrics)} (Friedman p<{ALPHA})."
            )
        if flat_metrics:
            bullets.append(
                f"- **{model} / {exp}**: invariant on {', '.join(flat_metrics)} "
                f"(Friedman p≥{ALPHA})."
            )

    # 2) Cross-model
    cm = results[results.test == "cross_model_delta_wilcoxon"]
    sig_cm = cm[cm.significant]
    if not sig_cm.empty:
        bullets.append("")
        bullets.append("**Cross-model significant Δ-comparisons** "
                       "(models differ in how they respond to input reduction):")
        for _, r in sig_cm.iterrows():
            bullets.append(
                f"- {r['scope']} / {r['metric']}: Δ_main = {r['model_a_mean_delta']:+.4g}, "
                f"Δ_other = {r['model_b_mean_delta']:+.4g}, "
                f"p={r['p_value']:.3g} {r['sig_marker']}"
            )

    # 3) Spearman
    sp = results[results.test == "spearman_per_subject"]
    sig_sp = sp[(sp.significant) & (sp.scope == "all")]
    if not sig_sp.empty:
        bullets.append("")
        bullets.append("**Significant per-subject Spearman correlations (scope=all):**")
        for _, r in sig_sp.iterrows():
            bullets.append(
                f"- {r['model']} / {r['experiment']} / {r['metric']}: "
                f"mean ρ = {r['mean_rho']:+.3f}, p = {r['p_value']:.3g} {r['sig_marker']}"
            )

    return bullets


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_to_wandb(results, summary_path):
    try:
        import wandb
    except ImportError:
        print("  [skip] wandb not installed")
        return
    try:
        run = wandb.init(
            project="SYNBADD_Experiments_Robustness_Stats",
            name="significance_tests_fold2",
            config={"alpha": ALPHA, "fold": 2},
            reinit=True,
        )
    except Exception as e:
        print(f"  [skip] wandb init failed: {e}")
        return

    # Full table
    run.log({"all_tests": wandb.Table(dataframe=results)})

    # Per-test-type tables
    for test_name, sub in results.groupby("test"):
        run.log({f"{test_name}_table": wandb.Table(dataframe=sub)})

    # Summary scalars: per (model, experiment) — fraction of significant Friedman tests
    fried = results[results.test == "friedman_within_subject"]
    for (model, exp), sub in fried.groupby(["model", "experiment"]):
        run.summary[f"{model}/{exp}/frac_friedman_sig"] = float(sub.significant.mean())
        run.summary[f"{model}/{exp}/n_friedman_tests"] = int(len(sub))

    cm = results[results.test == "cross_model_delta_wilcoxon"]
    if not cm.empty:
        run.summary["cross_model/frac_significant"] = float(cm.significant.mean())
        run.summary["cross_model/n_tests"] = int(len(cm))

    # Markdown report as artifact + as a text panel
    if os.path.exists(summary_path):
        run.log({"report_md": wandb.Html("<pre>" + Path(summary_path).read_text() + "</pre>")})

    run.finish()
    print(f"  [wandb] logged to project SYNBADD_Experiments_Robustness_Stats")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=> output dir: {args.output_dir}")

    print("=> loading data")
    data = load_data()
    if not data:
        print("No data found.")
        return

    print(f"=> running tests over {len(data)} (model, experiment) cells")
    results = collect_tests(data)
    print(f"   produced {len(results)} test rows")

    csv_path = os.path.join(args.output_dir, "significance_tests.csv")
    results.to_csv(csv_path, index=False)
    print(f"=> CSV : {csv_path}")

    md_path = os.path.join(args.output_dir, "REPORT.md")
    write_markdown(results, md_path)
    print(f"=> MD  : {md_path}")

    if not args.no_wandb:
        print("=> logging to W&B")
        log_to_wandb(results, md_path)

    # Quick stdout summary
    print("\n=== quick summary ===")
    print(f"  total tests:           {len(results)}")
    print(f"  significant (p<{ALPHA}): {int(results.significant.sum())}")
    print(f"  by test type:")
    for t, sub in results.groupby("test"):
        print(f"    {t:<32} n={len(sub):<4} sig={int(sub.significant.sum())}")


if __name__ == "__main__":
    main()
