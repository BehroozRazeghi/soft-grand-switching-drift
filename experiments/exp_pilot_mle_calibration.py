from __future__ import annotations
import csv
import math
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ggn import sample_ggn, constrained_mle

SEED = 2026081912
REPS = 64
M_VALUES = (20, 25, 30, 40, 50, 60, 75, 100, 120, 150)
DATA_DIR = ROOT / "data"
PARAMS = ((1.08, -0.28), (1.08, 0.28), (1.50, 0.0), (1.92, -0.28), (1.92, 0.28))
BETA_BOUNDS = (1.02, 1.98)
ETA_BOUNDS = (math.log(0.65), math.log(1.45))
BOUNDARY_TOL = 5e-8


def mean_ci95(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    return m, float(student_t.ppf(0.975, len(x) - 1) * se)


def _boundary_hit(est):
    beta, eta = map(float, est)
    return (
        abs(beta - BETA_BOUNDS[0]) <= BOUNDARY_TOL
        or abs(beta - BETA_BOUNDS[1]) <= BOUNDARY_TOL
        or abs(eta - ETA_BOUNDS[0]) <= BOUNDARY_TOL
        or abs(eta - ETA_BOUNDS[1]) <= BOUNDARY_TOL
    )


def calibration_rng(param_index: int, m: int, rep: int) -> np.random.Generator:
    """Return a distinct deterministic stream initialization for one fit."""
    return np.random.default_rng(
        np.random.SeedSequence([SEED, int(param_index), int(m), int(rep)])
    )


def run(write=True):
    rows = []
    C_hat = 0.0
    total_hits = 0
    total_fits = 0
    for pi, theta in enumerate(PARAMS):
        target = np.asarray(theta, float)
        for m in M_VALUES:
            errs = []
            hits = 0
            for rep in range(REPS):
                rng = calibration_rng(pi, m, rep)
                z = sample_ggn(theta[0], theta[1], m, rng)
                est, _ = constrained_mle(z, BETA_BOUNDS, ETA_BOUNDS)
                errs.append(float(np.linalg.norm(est - target)))
                hits += int(_boundary_hit(est))
            mean, ci = mean_ci95(errs)
            scaled = math.sqrt(m) * mean
            scaled_upper = math.sqrt(m) * (mean + ci)
            C_hat = max(C_hat, scaled_upper)
            total_hits += hits
            total_fits += REPS
            rows.append({
                "beta": theta[0],
                "eta": theta[1],
                "m": m,
                "mean_error": mean,
                "ci95": ci,
                "sqrt_m_mean_error": scaled,
                "sqrt_m_upper95": scaled_upper,
                "boundary_hit_fraction": hits / REPS,
            })
    meta = {
        "seed": SEED,
        "rng_scheme": "SeedSequence([seed,param_index,m,rep])",
        "reps": REPS,
        "C_est_calibrated_upper95": C_hat,
        "calibration_grid_points": len(PARAMS) * len(M_VALUES),
        "boundary_hit_fraction_all_fits": total_hits / total_fits,
    }
    if write:
        with (DATA_DIR / "pilot_mle_calibration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (DATA_DIR / "pilot_mle_calibration_meta.txt").write_text(
            "\n".join(
                f"{k}={v:.17g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in meta.items()
            ) + "\n"
        )
    return rows, meta


if __name__ == "__main__":
    rows, meta = run(True)
    print(meta)
