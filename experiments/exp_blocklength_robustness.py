from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_pilot_drift import (
    B,
    BETA_BOUNDS,
    ETA_BOUNDS,
    GAMMA,
    PILOT_TO_DATA_RATIO,
    R,
    theta_path,
)
from src.decoding import (
    bit_matrix,
    error_for_log_order_batch,
    error_regret_for_order_batch,
    grandab_kernel_by_rank,
    prepare_true_posterior_batch,
)
from src.ggn import constrained_mle, sample_ggn
from src.pilot_tracking import hold_last_refresh

SEED = 2026081915
REPS = 8
N_VALUES = (8, 10, 12)
CASES = ((120, 20), (40, 8))
DATA_DIR = ROOT / "data"


def run(write: bool = True):
    rows = []
    for period, q in CASES:
        theta = theta_path(period)
        for n in N_VALUES:
            bits = bit_matrix(n)
            ambient_size = 1 << n
            codebook_size = int(math.floor(2 ** (n * R)))
            query_budget = int(math.floor(2 ** (n * GAMMA)))
            kernel = grandab_kernel_by_rank(
                ambient_size, codebook_size, query_budget
            )
            m = int(round(PILOT_TO_DATA_RATIO * n * q))
            excess_errors = []

            for rep in range(REPS):
                data_rng = np.random.default_rng(
                    SEED + period * 10_000 + n * 100 + rep
                )
                observations = []
                for b in range(B):
                    x = data_rng.choice(np.array([-1.0, 1.0]), size=n)
                    observations.append(
                        x + sample_ggn(theta[b, 0], theta[b, 1], n, data_rng)
                    )
                observations = np.stack(observations)

                log_posterior, posterior = prepare_true_posterior_batch(
                    observations, theta, bits
                )
                oracle_error = error_for_log_order_batch(
                    log_posterior, posterior, kernel
                )

                pilot_rng = np.random.default_rng(
                    SEED + 1_000_000 + period * 10_000 + n * 100 + rep
                )
                estimates = {}
                for start in range(0, B, q):
                    residuals = sample_ggn(
                        theta[start, 0], theta[start, 1], m, pilot_rng
                    )
                    estimate, _ = constrained_mle(
                        residuals, BETA_BOUNDS, ETA_BOUNDS
                    )
                    estimates[start] = estimate

                tracked_theta = hold_last_refresh(theta, estimates, q)
                tracked_error, _ = error_regret_for_order_batch(
                    observations,
                    tracked_theta,
                    bits,
                    kernel,
                    log_posterior,
                    posterior,
                )
                excess_errors.append(
                    float(np.mean(tracked_error - oracle_error))
                )

            rows.append(
                {
                    "period": period,
                    "q": q,
                    "n": n,
                    "m": m,
                    "reps": REPS,
                    "mean_excess_error": float(np.mean(excess_errors)),
                    "se": float(
                        np.std(excess_errors, ddof=1) / math.sqrt(REPS)
                    ),
                }
            )

    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (DATA_DIR / "blocklength_robustness.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    return rows


if __name__ == "__main__":
    for row in run(True):
        print(row)
