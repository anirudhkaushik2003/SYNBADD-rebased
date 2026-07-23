"""Shared plotting style + data-loading helpers for SynBADD paper figures.

Each individual figure script imports from here so styling is consistent.
Edit this file to change fonts, colors, sizes, etc. globally.
"""

from pathlib import Path

import matplotlib as mpl
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
IMG_DIR = REPO / "paper" / "images"
QUAL = Path("/scratch/anirudhkaushik/SYNBADD_outputs/SYNBADD_Qualitative_2")

EXP_DIRS = {
    ("main", "exp1"): QUAL / "experiment_reduced_time_input_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2",
    ("nounet", "exp1"): QUAL / "experiment_reduced_time_input_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2",
    ("main", "exp2"): QUAL / "experiment_asymmetric_information_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2",
    ("nounet", "exp2"): QUAL / "experiment_asymmetric_information_fold_2"
        / "SYNBADD_128_6_5e-05_0.0_2025-05-28_15-15-06_ADNI_all_longitudinal_CrossAttentionTransformer+NoUnet_Past+Future_Fold2",
}

# ---------------------------------------------------------------------------
# Labels and colors
# ---------------------------------------------------------------------------

MODEL_LABEL = {
    "main":   r"UNet$+$DAFT (full)",
    "nounet": r"No UNet$+$No DAFT",
}
MODEL_COLOR = {"main": "#0072B2", "nounet": "#D55E00"}
GROUP_COLOR = {"CN": "#009E73", "MCI": "#E69F00", "AD": "#CC79A7"}

REGIONS_ORDER = [
    "hippocampus", "amygdala", "entorhinal_cortex", "parahippocampal",
    "fusiform_gyrus", "inferior_temporal_gyrus", "middle_temporal_gyrus",
    "third_ventricle",
]
REGION_NICE = {
    "hippocampus": "Hippocampus",
    "amygdala": "Amygdala",
    "entorhinal_cortex": "Entorhinal Cortex",
    "parahippocampal": "Parahippocampal",
    "fusiform_gyrus": "Fusiform Gyrus",
    "inferior_temporal_gyrus": "Inferior Temporal G.",
    "middle_temporal_gyrus": "Middle Temporal G.",
    "third_ventricle": "Third Ventricle",
}

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def apply_style():
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
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(model: str, exp: str):
    """Return (per_config_metrics_df, per_config_dice_df_or_None)."""
    d = EXP_DIRS[(model, exp)]
    per = pd.read_csv(d / "per_config_metrics.csv")
    dice_path = d / "per_config_dice_volume.csv"
    dice = pd.read_csv(dice_path) if dice_path.exists() else None
    return per, dice


def save_pdf(fig, name: str) -> Path:
    """Save fig to paper/images/{name}.pdf."""
    IMG_DIR.mkdir(exist_ok=True)
    out = IMG_DIR / f"{name}.pdf"
    fig.savefig(out)
    return out
