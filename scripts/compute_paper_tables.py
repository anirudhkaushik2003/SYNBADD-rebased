#!/usr/bin/env python3
"""Aggregate per-fold metrics for the SynBADD MedIA paper tables.

Produces:
  1. per-fold image quality (PSNR/SSIM/MSE/NRMSE) for the main model (5 folds)
  2. per-fold Jacobian determinant statistics (mean, std)               (5 folds)
  3. per-(target timepoint) image quality metrics                        (5 folds)
  4. per-(diagnostic group) image quality + Jacobian metrics            (folds with group cols)

Outputs CSV summaries to paper/tables/ and prints LaTeX-ready snippets.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/anirudhkaushik/SYNBADD-rebased")
OUT_DIR = REPO / "paper" / "tables"
OUT_DIR.mkdir(exist_ok=True)

PROJ = "/scratch/anirudhkaushik/SYNBADD/SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations/test"


def latest_per_fold(must_contain="_CrossAttentionTransformer+DAFT_",
                    require_group_cols=False):
    runs = {}
    for f in glob.glob(f"{PROJ}/*Fold*_test.csv"):
        if must_contain not in f or "GatedCrossAttention" in f:
            continue
        m = re.search(r"Fold(\d+)", f)
        d = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", f)
        if not m or not d:
            continue
        if require_group_cols:
            try:
                df = pd.read_csv(f, nrows=1)
                if "Test SSIM CN" not in df.columns:
                    continue
            except Exception:
                continue
        fold = int(m.group(1))
        date = d.group(1) + d.group(2)
        cur = runs.get(fold)
        if cur is None or date > cur[0]:
            runs[fold] = (date, f)
    return {fold: path for fold, (_, path) in runs.items()}


def main():
    # ---- 5-fold runs (overall metrics) ----
    files = latest_per_fold()
    print("=== Latest CrossAttentionTransformer+DAFT per fold ===")
    for fold in sorted(files):
        print(f"  fold {fold}: {Path(files[fold]).name}")

    rows = []
    for fold in sorted(files):
        df = pd.read_csv(files[fold])
        r = df.iloc[0]
        row = {
            "fold": fold,
            "psnr": r["Test PSNR"], "ssim": r["Test SSIM"],
            "mssim": r["Test MS_SSIM"], "mse": r["Test MSE"],
            "nrmse": r["Test NRMSE"],
            "mean_jac": r["Test Mean Jacobian"], "std_jac": r["Test Std Jacobian"],
        }
        for tp in ["m06", "m12", "m18", "m24", "m36"]:
            row[f"psnr_{tp}"] = r[f"Test PSNR {tp}"]
            row[f"ssim_{tp}"] = r[f"Test SSIM {tp}"]
            row[f"mean_jac_{tp}"] = r[f"Test Mean Jacobian {tp}"]
        rows.append(row)
    perfold = pd.DataFrame(rows).set_index("fold")
    summary_row = perfold.mean(axis=0).to_frame().T
    summary_row.index = ["mean"]
    perfold = pd.concat([perfold, summary_row])
    perfold.to_csv(OUT_DIR / "perfold_main_metrics.csv")

    print("\n=== per-fold summary ===")
    print(perfold[["psnr", "ssim", "mse", "nrmse", "mean_jac", "std_jac"]].round(4).to_string())

    print("\n---- Per-fold PSNR/SSIM/Jacobian LaTeX rows ----")
    for fold in [1, 2, 3, 4, 5]:
        r = perfold.loc[fold]
        print(f"Fold {fold} & {r.psnr:.4f} & {r.ssim:.4f} & {r.mean_jac:.4f} & {r.std_jac:.4f} \\\\")
    r = perfold.loc["mean"]
    print(f"\\textbf{{Mean}} & \\textbf{{{r.psnr:.4f}}} & \\textbf{{{r.ssim:.4f}}} & \\textbf{{{r.mean_jac:.4f}}} & \\textbf{{{r.std_jac:.4f}}} \\\\")

    print("\n---- Per-target-timepoint averages over 5 folds ----")
    for tp in ["m06", "m12", "m18", "m24", "m36"]:
        psnr_m = perfold.loc[1:5, f"psnr_{tp}"].mean()
        ssim_m = perfold.loc[1:5, f"ssim_{tp}"].mean()
        jac_m = perfold.loc[1:5, f"mean_jac_{tp}"].mean()
        print(f"  {tp}: PSNR={psnr_m:.4f}  SSIM={ssim_m:.4f}  mean Jac={jac_m:.4f}")
    print("\nLaTeX rows for per-timepoint:")
    for tp in ["m06", "m12", "m18", "m24", "m36"]:
        psnr_m = perfold.loc[1:5, f"psnr_{tp}"].mean()
        ssim_m = perfold.loc[1:5, f"ssim_{tp}"].mean()
        jac_m = perfold.loc[1:5, f"mean_jac_{tp}"].mean()
        print(f"{tp.upper()} & {psnr_m:.4f} & {ssim_m:.4f} & {jac_m:.4f} \\\\")

    # ---- subgroup runs (folds with group columns) ----
    print("\n=== Folds with subgroup CN/MCI/AD columns ===")
    grp_files = latest_per_fold(require_group_cols=True)
    for fold in sorted(grp_files):
        print(f"  fold {fold}: {Path(grp_files[fold]).name}")

    grp_rows = []
    for fold in sorted(grp_files):
        df = pd.read_csv(grp_files[fold])
        r = df.iloc[0]
        for grp in ["CN", "MCI", "AD"]:
            grp_rows.append({
                "fold": fold,
                "group": grp,
                "psnr": r[f"Test PSNR {grp}"],
                "ssim": r[f"Test SSIM {grp}"],
                "mse": r[f"Test MSE {grp}"],
                "nrmse": r[f"Test NRMSE {grp}"],
                "mean_jac": r[f"Test Mean Jacobian {grp}"],
                "std_jac": r[f"Test Std Jacobian {grp}"],
            })
    grp_df = pd.DataFrame(grp_rows)
    if not grp_df.empty:
        grp_df.to_csv(OUT_DIR / "subgroup_metrics.csv", index=False)
        print("\n=== Subgroup summary (mean across the available folds) ===")
        avg = grp_df.groupby("group")[["psnr", "ssim", "mse", "nrmse", "mean_jac", "std_jac"]].mean().round(4)
        print(avg.to_string())
        print("\nLaTeX rows:")
        for grp in ["CN", "MCI", "AD"]:
            row = avg.loc[grp]
            print(f"{grp} & {row.psnr:.4f} & {row.ssim:.4f} & {row.mean_jac:.4f} & {row.std_jac:.4f} \\\\")


if __name__ == "__main__":
    main()
