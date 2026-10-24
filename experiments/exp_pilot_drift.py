from __future__ import annotations
import csv
import math
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decoding import (
    bit_matrix,
    grandab_kernel_by_rank,
    prepare_true_posterior_batch,
    error_regret_for_order_batch,
    error_for_log_order_batch,
)
from src.ggn import sample_ggn, constrained_mle, conditional_entropy_bpsk_ggn
from src.pilot_tracking import finite_tracking_bound, hold_last_refresh, path_length

SEED = 2026081914
N_DATA = 10
B = 120
PILOT_TO_DATA_RATIO = 0.50
Q_VALUES = (4, 5, 6, 8, 10, 12, 15, 20, 24, 30)
PERIODS = (120, 40)
DATA_DIR = ROOT / "data"
REPS = 64
R = 0.25
GAMMA = 0.65
BETA_AMP = 0.42
ETA_AMP = 0.28
ETA_PHASE = 0.7
BETA_BOUNDS = (1.02, 1.98)
ETA_BOUNDS = (math.log(0.65), math.log(1.45))
BOUNDARY_TOL = 5e-8


def theta_path(period: int, B0: int = B):
    # Match the manuscript convention b=1,...,B exactly.
    b = np.arange(1, B0 + 1, dtype=float)
    beta = 1.50 + BETA_AMP * np.sin(2 * np.pi * b / period - np.pi / 2)
    eta = ETA_AMP * np.sin(2 * np.pi * b / period - np.pi / 2 + ETA_PHASE)
    return np.column_stack([beta, eta])


def simulate_data_block(theta, rng, n=N_DATA):
    beta, eta = theta
    x = rng.choice(np.array([-1.0, 1.0]), size=n)
    z = sample_ggn(beta, eta, n, rng)
    return x + z


def mean_se_ci95(values):
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if len(x) <= 1:
        return mean, 0.0, 0.0
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    half = float(student_t.ppf(0.975, df=len(x) - 1) * se)
    return mean, se, half


def max_true_path_entropy(th):
    """Return max_b H(X|Y; theta_b) along the specified true path."""
    vals = np.array([conditional_entropy_bpsk_ggn(float(b), float(e)) for b, e in th])
    idx = int(np.argmax(vals))
    return float(vals[idx]), idx


def _boundary_hit(est):
    beta, eta = map(float, est)
    return (
        abs(beta - BETA_BOUNDS[0]) <= BOUNDARY_TOL
        or abs(beta - BETA_BOUNDS[1]) <= BOUNDARY_TOL
        or abs(eta - ETA_BOUNDS[0]) <= BOUNDARY_TOL
        or abs(eta - ETA_BOUNDS[1]) <= BOUNDARY_TOL
    )


def estimate_held_path(th, q, m, seed_base):
    ests = {}
    no_refinement = 0
    boundary_hits = 0
    rng = np.random.default_rng(seed_base)
    for s in range(0, len(th), q):
        zpilot = sample_ggn(th[s, 0], th[s, 1], m, rng)
        est, refinement_succeeded = constrained_mle(zpilot, BETA_BOUNDS, ETA_BOUNDS)
        no_refinement += int(not refinement_succeeded)
        boundary_hits += int(_boundary_hit(est))
        ests[s] = est
    return hold_last_refresh(th, ests, q), ests, no_refinement, boundary_hits


def _run_direct(write: bool = True):
    bits = bit_matrix(N_DATA)
    N = 1 << N_DATA
    M = int(math.floor(2 ** (N_DATA * R)))
    tau = int(math.floor(2 ** (N_DATA * GAMMA)))
    kernel = grandab_kernel_by_rank(N, M, tau)
    rows = []
    metadata = {
        "seed": SEED,
        "B": B,
        "n": N_DATA,
        "pilot_to_data_ratio": PILOT_TO_DATA_RATIO,
        "reps": REPS,
        "R": R,
        "gamma": GAMMA,
        "N": N,
        "M": M,
        "tau": tau,
    }
    total_no_refinement = 0
    total_boundary_hits = 0
    total_mle_fits = 0

    for period in PERIODS:
        th = theta_path(period)
        V = path_length(th)
        h_path_max, h_path_max_index = max_true_path_entropy(th)
        if not (h_path_max < GAMMA < 1.0 - R):
            raise RuntimeError(
                "finite-path entropy/budget check failed: "
                f"period={period} {h_path_max=} {GAMMA=} {1-R=}"
            )
        for q in Q_VALUES:
            if B % q:
                raise RuntimeError(f"B={B} must be divisible by q={q}")

        init_theta = th[0].copy()
        oracle = []
        initial_held = []
        accum = {
            q: {"tracked_raw": [], "track": [], "bound": [], "regret": []}
            for q in Q_VALUES
        }

        # The same data realization is used for the oracle, stale baseline, and
        # every q within a repetition; only pilot observations differ by q.
        for rep in range(REPS):
            rngd = np.random.default_rng(SEED + 1_000_000 * period + rep)
            blocks = np.stack([simulate_data_block(th[b], rngd) for b in range(B)], axis=0)
            lpo, po = prepare_true_posterior_batch(blocks, th, bits)
            eo = error_for_log_order_batch(lpo, po, kernel)
            init_path = np.repeat(init_theta[None, :], B, axis=0)
            ei, _ = error_regret_for_order_batch(blocks, init_path, bits, kernel, lpo, po)
            oracle.append(float(np.mean(eo)))
            initial_held.append(float(np.mean(ei)))

            for q in Q_VALUES:
                m = int(round(PILOT_TO_DATA_RATIO * N_DATA * q))
                held, ests, no_refine, hits = estimate_held_path(
                    th,
                    q,
                    m,
                    SEED + 10_000_000 * period + 10_000 * q + rep,
                )
                total_no_refinement += no_refine
                total_boundary_hits += hits
                total_mle_fits += len(ests)
                actual, bound = finite_tracking_bound(th, ests, q)
                et, rg = error_regret_for_order_batch(blocks, held, bits, kernel, lpo, po)
                a = accum[q]
                a["tracked_raw"].append(float(np.mean(et)))
                a["track"].append(actual)
                a["bound"].append(bound)
                a["regret"].append(float(np.mean(rg)))

        init_diff = np.asarray(initial_held) - np.asarray(oracle)
        init_mean, init_se, init_ci = mean_se_ci95(init_diff)

        for q in Q_VALUES:
            m = int(round(PILOT_TO_DATA_RATIO * N_DATA * q))
            actual_ratio = (B // q) * m / (B * N_DATA)
            a = accum[q]
            tracked_raw = a["tracked_raw"]
            tracked_diff = np.asarray(tracked_raw) - np.asarray(oracle)
            tr_mean, tr_se, tr_ci = mean_se_ci95(tracked_diff)
            te_mean, te_se, _ = mean_se_ci95(a["track"])
            bd_mean, bd_se, _ = mean_se_ci95(a["bound"])
            rg_mean, rg_se, _ = mean_se_ci95(a["regret"])
            or_mean, _, _ = mean_se_ci95(oracle)
            rows.append({
                "period": period,
                "v": V / B,
                "q": q,
                "m": m,
                "pilot_to_data_ratio_nominal": PILOT_TO_DATA_RATIO,
                "pilot_to_data_ratio_actual": actual_ratio,
                "avg_tracking_error": te_mean,
                "tracking_error_se": te_se,
                "tracking_bound": bd_mean,
                "tracking_bound_se": bd_se,
                "D_plus_bits_per_symbol": rg_mean,
                "D_plus_se": rg_se,
                "grandab_error_oracle": or_mean,
                "grandab_error_tracked": float(np.mean(tracked_raw)),
                "grandab_error_true_initial_held": float(np.mean(initial_held)),
                "tracked_excess_mean": tr_mean,
                "tracked_excess_se": tr_se,
                "tracked_excess_ci95": tr_ci,
                "true_initial_held_excess_mean": init_mean,
                "true_initial_held_excess_se": init_se,
                "true_initial_held_excess_ci95": init_ci,
                "h_max_true_path": h_path_max,
            })
        metadata[f"period_{period}_V_B"] = V
        metadata[f"period_{period}_v"] = V / B
        metadata[f"period_{period}_h_max_true_path"] = h_path_max
        metadata[f"period_{period}_h_max_true_path_index"] = h_path_max_index
    metadata["mle_no_refinement_flags"] = total_no_refinement
    metadata["mle_boundary_hit_fraction"] = (
        total_boundary_hits / total_mle_fits if total_mle_fits else 0.0
    )
    metadata["mle_fit_count"] = total_mle_fits

    if write:
        out = DATA_DIR / "pilot_drift.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (DATA_DIR / "pilot_drift_meta.txt").write_text(
            "\n".join(
                f"{k}={v:.17g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metadata.items()
            ) + "\n"
        )
    return rows, metadata



def run(write: bool = True):
    return _run_direct(write=write)


if __name__ == "__main__":
    rows, meta = run(True)
    print(meta)
    for row in rows:
        print(row)
