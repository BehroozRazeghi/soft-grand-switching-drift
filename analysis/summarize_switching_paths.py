from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Summarize all-path switching results.")
    ap.add_argument("--input", type=Path, required=True, help="switching_channel_pathwise.csv")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    with args.input.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for S in sorted({int(r["S"]) for r in rows}):
        rr = [r for r in rows if int(r["S"]) == S]
        mix = np.array([float(r["switching_excess_mean"]) for r in rr])
        sta = np.array([float(r["initial_state_static_excess_mean"]) for r in rr])
        diff = sta - mix
        out.append({
            "S": S,
            "path_count": len(rr),
            "switching_mean": float(np.mean(mix)),
            "switching_q05": float(np.quantile(mix, 0.05)),
            "switching_median": float(np.median(mix)),
            "switching_q95": float(np.quantile(mix, 0.95)),
            "static_mean": float(np.mean(sta)),
            "static_q05": float(np.quantile(sta, 0.05)),
            "static_median": float(np.median(sta)),
            "static_q95": float(np.quantile(sta, 0.95)),
            "fraction_paths_switching_lower": float(np.mean(mix < sta)),
            "mean_static_minus_switching": float(np.mean(diff)),
        })
    output = args.output or args.input.with_name("switching_path_summary_extended.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(output)


if __name__ == "__main__":
    main()
