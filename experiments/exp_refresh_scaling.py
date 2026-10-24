from __future__ import annotations
import csv
import math
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ggn import sample_ggn, constrained_mle
from src.pilot_tracking import path_length, finite_tracking_bound, refresh_optimum
from experiments.exp_pilot_mle_calibration import BETA_BOUNDS, ETA_BOUNDS

SEED = 2026081913
B = 120
N_DATA = 10
PHI = 0.50
PERIODS = (120, 80, 60, 40, 30)
Q_VALUES = (4, 5, 6, 8, 10, 12, 15, 20, 24, 30)
REPS = 64
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2026082020
MAX_WORKERS = min(8, os.cpu_count() or 1)
DATA_DIR = ROOT / "data"
BETA_AMP = 0.42
ETA_AMP = 0.28
ETA_PHASE = 0.7


def theta_path(period):
    # Manuscript block indexing is b=1,...,B; use the same indexing here.
    b = np.arange(1, B + 1, dtype=float)
    return np.column_stack([
        1.50 + BETA_AMP * np.sin(2 * np.pi * b / period - np.pi / 2),
        ETA_AMP * np.sin(2 * np.pi * b / period - np.pi / 2 + ETA_PHASE),
    ])


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


def _evaluate_period_q(task):
    period, q, C = task
    th = theta_path(period)
    v = path_length(th) / B
    q_star, _ = refresh_optimum(C, v, PHI, N_DATA)
    if B % q:
        raise RuntimeError("q must divide B")
    m = int(round(PHI * N_DATA * q))
    vals = []
    for rep in range(REPS):
        rng = np.random.default_rng(SEED + 100000 * period + 1000 * q + rep)
        ests = {}
        for s in range(0, B, q):
            z = sample_ggn(th[s, 0], th[s, 1], m, rng)
            est, _ = constrained_mle(z, BETA_BOUNDS, ETA_BOUNDS)
            ests[s] = est
        actual, bound = finite_tracking_bound(th, ests, q)
        vals.append((actual, bound))
    arr = np.asarray(vals)
    mean, se = _mean_se(arr[:, 0])
    bmean, bse = _mean_se(arr[:, 1])
    tracking_bound_f_q = C / math.sqrt(PHI * N_DATA * q) + q * v
    return {
        "_raw_tracking": arr[:, 0].tolist(),
        "_raw_bound": arr[:, 1].tolist(),
        "period": period,
        "v": v,
        "q": q,
        "m": m,
        "reps": REPS,
        "tracking_mean": mean,
        "tracking_se": se,
        "finite_bound_mean": bmean,
        "finite_bound_se": bse,
        "tracking_bound_f_q": tracking_bound_f_q,
        "q_star": q_star,
    }


def _read_calibrated_C():
    meta_path = DATA_DIR / "pilot_mle_calibration_meta.txt"
    if not meta_path.is_file():
        raise FileNotFoundError("run exp_pilot_mle_calibration.py before exp_refresh_scaling.py")
    vals = {}
    for line in meta_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    return float(vals["C_est_calibrated_upper95"])


def _bootstrap_grid_argmin(period_rows):
    """Independent-within-q bootstrap of the descriptive grid argmin."""
    ordered = sorted(period_rows, key=lambda r: int(r["q"]))
    q_grid = np.asarray([int(r["q"]) for r in ordered], dtype=int)
    period = int(ordered[0]["period"])
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_SEED, period])
    )
    bootstrap_means = np.empty((len(ordered), BOOTSTRAP_REPS), dtype=float)
    for j, row in enumerate(ordered):
        values = np.asarray(row["_raw_tracking"], dtype=float)
        indices = rng.integers(
            0, len(values), size=(BOOTSTRAP_REPS, len(values))
        )
        bootstrap_means[j] = np.mean(values[indices], axis=1)
    selected = q_grid[np.argmin(bootstrap_means, axis=0)]
    low, high = np.quantile(selected, (0.025, 0.975), method="nearest")
    values, counts = np.unique(selected, return_counts=True)
    mode = int(values[np.argmax(counts)])
    return selected, int(low), int(high), mode


def run(write=True, return_raw=False):
    C = _read_calibrated_C()
    tasks = [(period, q, C) for period in PERIODS for q in Q_VALUES]
    workers = MAX_WORKERS
    if workers <= 1:
        rows = [_evaluate_period_q(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_evaluate_period_q, tasks))
    rows.sort(key=lambda r: (int(r["period"]), int(r["q"])))
    raw_rows = []
    for r in rows:
        for rep, (track, bnd) in enumerate(
            zip(r["_raw_tracking"], r["_raw_bound"])
        ):
            raw_rows.append({
                "period": r["period"],
                "v": r["v"],
                "q": r["q"],
                "m": r["m"],
                "rep": rep,
                "tracking_error": track,
                "finite_bound": bnd,
            })
    summary = []
    for period in PERIODS:
        pr = [r for r in rows if int(r["period"]) == period]
        v = float(pr[0]["v"])
        q_star = float(pr[0]["q_star"])
        q_emp = min((float(r["tracking_mean"]), int(r["q"])) for r in pr)[1]
        q_nearest = min(Q_VALUES, key=lambda q: (abs(q - q_star), q))
        selected, q_low, q_high, q_mode = _bootstrap_grid_argmin(pr)
        summary.append({
            "period": period,
            "v": v,
            "q_empirical": q_emp,
            "q_nearest_to_q_star": q_nearest,
            "q_star": q_star,
            "C_est": C,
            "reps": REPS,
            "q_empirical_bootstrap_low95": q_low,
            "q_empirical_bootstrap_high95": q_high,
            "q_empirical_bootstrap_mode": q_mode,
            "bootstrap_probability_at_q_empirical": float(np.mean(selected == q_emp)),
            "bootstrap_reps": BOOTSTRAP_REPS,
        })
    for r in rows:
        r.pop("_raw_tracking", None)
        r.pop("_raw_bound", None)
    if write:
        with (DATA_DIR / "refresh_scaling.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        with (DATA_DIR / "refresh_scaling_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)
        with (DATA_DIR / "refresh_scaling_raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
            w.writeheader(); w.writerows(raw_rows)
    if return_raw:
        return rows, summary, raw_rows
    return rows, summary


if __name__ == "__main__":
    _, s = run(True)
    for r in s:
        print(r)
