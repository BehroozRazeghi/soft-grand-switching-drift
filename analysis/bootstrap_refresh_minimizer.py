from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Bootstrap uncertainty of the empirical refresh-grid minimizer.")
    ap.add_argument("--input", type=Path, required=True, help="refresh_scaling_raw.csv")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026082020)
    args = ap.parse_args()

    rows = read_rows(args.input)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[int(r["period"])][int(r["q"])].append(float(r["tracking_error"]))

    out = []
    for period in sorted(grouped, reverse=True):
        qmap = grouped[period]
        qvals = sorted(qmap)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, period]))
        bootstrap_means = np.empty((len(qvals), args.bootstrap), dtype=float)
        for j, q in enumerate(qvals):
            x = np.asarray(qmap[q], dtype=float)
            indices = rng.integers(0, len(x), size=(args.bootstrap, len(x)))
            bootstrap_means[j] = np.mean(x[indices], axis=1)
        selected = np.asarray(qvals, dtype=int)[np.argmin(bootstrap_means, axis=0)]
        counts = {q: int(np.sum(selected == q)) for q in qvals}
        for q in qvals:
            out.append({
                "period": period,
                "q": q,
                "bootstrap_minimizer_probability": counts[q] / args.bootstrap,
                "bootstrap_replicates": args.bootstrap,
            })

    output = args.output or args.input.with_name("refresh_minimizer_bootstrap.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(output)


if __name__ == "__main__":
    main()
