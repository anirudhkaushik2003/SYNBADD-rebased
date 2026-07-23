#!/usr/bin/env python3
"""Run every paper-figure script in order. Each script is independently
runnable; this is just a convenience driver.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ORDER = [
    "exp1_image_metrics.py",
    "exp1_dice_curves.py",
    "exp1_per_region_bars.py",
    "exp1_cross_model_delta.py",
    "exp2_metrics_by_config.py",
    "exp2_recency_curve.py",
    "exp2_past_vs_future_scatter.py",
]


def main():
    sys.path.insert(0, str(HERE))
    for fn in ORDER:
        path = HERE / fn
        if not path.exists():
            print(f"  [skip] {fn} (missing)")
            continue
        print(f"== {fn} ==")
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
