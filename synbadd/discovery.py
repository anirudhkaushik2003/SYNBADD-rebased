"""Discover available pretrained checkpoints and completed inference outputs.

Parses SYNBADD run name format:
  SYNBADD_{img}_{bs}_{lr}_{wd}_{date}_{time}_{dataset}_{longitudinal|prev_only}_{specials}_Fold{N}

Where `specials` encodes the temporal model type and optional modifier joined by '+':
  e.g.  GatedCrossAttentionTransformer+DAFT
        CrossAttentionTransformer+OnlyUnet
        CrossAttentionTransformer+NoUnet
        CrossAttentionTransformer          (DAFT implied, no suffix)
        AveragedImputation
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Run-name regex
# ---------------------------------------------------------------------------
_RUN_RE = re.compile(
    r"^SYNBADD_(?P<img>\d+)_(?P<bs>\d+)_(?P<lr>[^_]+)_(?P<wd>[^_]+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<dataset>ADNI_all_(?:longitudinal|prev_only|future_only|cross_sectional))_"
    r"(?P<specials>[^_].*?)_"                     # e.g. CrossAttentionTransformer+DAFT
    r"(?P<mode>Past\+Future|OnlyPast|OnlyFuture|Past\.Future)_"
    r"Fold(?P<fold>\d+)$"
)

# Human-readable labels for temporal type keys
TEMPORAL_LABELS = {
    "GatedCrossAttentionTransformer": "GatedCrossAttention",
    "CrossAttentionGatedTransformer": "CrossAttentionGated",
    "CrossAttentionTransformer":      "CrossAttention",
    "AveragedImputation":             "AveragedImputation",
}

# Reverse: label → config type string used in YAML
TEMPORAL_CONFIG = {
    "GatedCrossAttentionTransformer": "GatedCrossAttentionTransformer",
    "CrossAttentionGatedTransformer": "CrossAttentionGatedTransformer",
    "CrossAttentionTransformer":      "CrossAttentionTransformer",
    "AveragedImputation":             "AveragedImputation",
}

# Modifier → (using_daft, using_unet)
MODIFIER_CONFIG = {
    "DAFT":     (True,  True),
    "OnlyUnet": (False, True),
    "NoUnet":   (False, False),
    None:       (True,  True),   # no modifier → same as DAFT
}


@dataclass
class RunInfo:
    run_name: str          # full run name string
    project: str           # checkpoint project directory name
    temporal: str          # e.g. "CrossAttentionTransformer"
    modifier: Optional[str]  # "DAFT", "OnlyUnet", "NoUnet", or None
    mode: str              # "Past+Future", "OnlyPast", etc.
    fold: int
    date: str              # "2025-05-22"
    time: str              # "13-44-20"
    has_checkpoint: bool = False
    inference_done: bool = False
    n_synthetics: int = 0

    @property
    def datetime_str(self) -> str:
        t = self.time.replace("-", ":")
        return f"{self.date} {t}"

    @property
    def specials(self) -> str:
        if self.modifier:
            return f"{self.temporal}+{self.modifier}"
        return self.temporal

    @property
    def label(self) -> str:
        """Human-readable model label."""
        mod = f"+{self.modifier}" if self.modifier else ""
        return f"{self.temporal}{mod}"


def parse_run_name(name: str, project: str = "") -> Optional[RunInfo]:
    """Parse a run name string into a RunInfo. Returns None if it doesn't match."""
    # Strip checkpoint suffixes
    for suffix in ("_SYNBADD_SVF.pth", "_vae_SYNBADD_SVF.pth",
                   "_transformer_SYNBADD_SVF.pth", "_mask"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    m = _RUN_RE.match(name)
    if not m:
        return None

    specials = m.group("specials")
    if "+" in specials:
        temporal, modifier = specials.split("+", 1)
    else:
        temporal, modifier = specials, None

    mode = m.group("mode").replace(".", "+")  # normalise "Past.Future" → "Past+Future"

    return RunInfo(
        run_name=name,
        project=project,
        temporal=temporal,
        modifier=modifier,
        mode=mode,
        fold=int(m.group("fold")),
        date=m.group("date"),
        time=m.group("time"),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(ckpt_root: str) -> List[RunInfo]:
    """Scan checkpoint root for pretrained model checkpoints."""
    results = []
    if not os.path.isdir(ckpt_root):
        return results

    for proj in sorted(os.listdir(ckpt_root)):
        proj_path = os.path.join(ckpt_root, proj)
        if not os.path.isdir(proj_path):
            continue
        for fname in os.listdir(proj_path):
            # Main checkpoint: ends with _SYNBADD_SVF.pth, not vae/transformer variant
            if not fname.endswith("_SYNBADD_SVF.pth"):
                continue
            if "_vae_SYNBADD_SVF.pth" in fname or "_transformer_SYNBADD_SVF.pth" in fname:
                continue
            info = parse_run_name(fname, project=proj)
            if info:
                info.has_checkpoint = True
                results.append(info)

    return results


def discover_inference_outputs(qualitative_base: str) -> List[RunInfo]:
    """Scan SYNBADD_Qualitative_{N} dirs for completed inference outputs."""
    results = []
    for fold in range(1, 6):
        base = os.path.join(qualitative_base, f"SYNBADD_Qualitative_{fold}")
        if not os.path.isdir(base):
            continue
        for dname in os.listdir(base):
            full = os.path.join(base, dname)
            if not os.path.isdir(full):
                continue
            # Skip utility dirs
            if dname in ("evaluation", "Results") or dname.startswith("Avg_impute") or \
               dname.startswith("FINAL_") or dname.endswith("_mask"):
                continue
            try:
                files = os.listdir(full)
            except OSError:
                continue
            n_syn = sum(
                1 for f in files
                if f.endswith("_synthetic.nii.gz") and "_field" not in f
            )
            if n_syn == 0:
                continue
            info = parse_run_name(dname)
            if info:
                info.inference_done = True
                info.n_synthetics = n_syn
                results.append(info)

    return results


def build_catalog(
    ckpt_root: str,
    qualitative_base: str,
) -> Dict[str, List[RunInfo]]:
    """
    Return a merged catalog keyed by (temporal, modifier, mode).

    Each value is a list of RunInfo with has_checkpoint / inference_done populated.
    """
    ckpt_runs = {r.run_name: r for r in discover_checkpoints(ckpt_root)}
    infer_runs = {r.run_name: r for r in discover_inference_outputs(qualitative_base)}

    # Merge
    all_run_names = set(ckpt_runs) | set(infer_runs)
    merged: Dict[str, RunInfo] = {}
    for rn in all_run_names:
        if rn in ckpt_runs and rn in infer_runs:
            r = ckpt_runs[rn]
            r.inference_done = True
            r.n_synthetics = infer_runs[rn].n_synthetics
            merged[rn] = r
        elif rn in ckpt_runs:
            merged[rn] = ckpt_runs[rn]
        else:
            merged[rn] = infer_runs[rn]

    # Group by (temporal, modifier, mode)
    catalog: Dict[Tuple, List[RunInfo]] = defaultdict(list)
    for r in merged.values():
        catalog[(r.temporal, r.modifier, r.mode)].append(r)

    # Sort each group by date then fold
    for key in catalog:
        catalog[key].sort(key=lambda r: (r.date, r.time, r.fold))

    return dict(catalog)


def print_catalog(catalog: Dict) -> None:
    """Pretty-print the catalog."""
    print("\n=== Available Models ===\n")
    for (temporal, modifier, mode), runs in sorted(catalog.items(), key=lambda x: (x[0][0], x[0][1] or "", x[0][2])):
        label = f"{temporal}+{modifier}" if modifier else temporal
        ckpt_folds = sorted(r.fold for r in runs if r.has_checkpoint)
        infer_folds = sorted(r.fold for r in runs if r.inference_done)
        projects = sorted({r.project for r in runs if r.project})
        dates = sorted({r.date for r in runs})
        print(f"  {label:50s}  mode={mode}")
        print(f"    projects : {', '.join(projects) or '(infer-only)'}")
        print(f"    dates    : {', '.join(dates)}")
        print(f"    ckpt     : folds {ckpt_folds}")
        print(f"    infer    : folds {infer_folds}")
        print()
