from __future__ import annotations
import csv
import math
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decoding import bit_matrix, grandab_kernel_by_rank
from src.ggn import sample_ggn, correction_bit_logposterior, conditional_entropy_bpsk_ggn
from src.rank import deterministic_order, ranks_from_order
from src.switching_mixture import (
    switching_mixture_log_scores_from_candidates,
    count_switch_paths,
    count_exact_switch_paths,
    enumerate_paths,
)

SEED = 2026081911
N_DATA = 10
REPS = 64
S_VALUES = (0, 1, 2, 3)
INITIAL_STATE = 0
R = 0.25
GAMMA = 0.65
MAX_WORKERS = min(8, os.cpu_count() or 1)
DATA_DIR = ROOT / "data"
STATE_THETA = np.array([[1.05, math.log(0.75)], [1.95, math.log(1.30)]], dtype=float)
MISMATCH_TOL = 2e-10


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    if len(x) <= 1:
        return mean, 0.0
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return mean, float(student_t.ppf(0.975, len(x) - 1) * se)


def candidate_logprob_tensor(y: np.ndarray, bits: np.ndarray) -> np.ndarray:
    N, n = bits.shape
    K = STATE_THETA.shape[0]
    out = np.empty((N, n, K), dtype=float)
    for k, (beta, eta) in enumerate(STATE_THETA):
        lp0, lp1 = correction_bit_logposterior(y, beta, eta)
        out[:, :, k] = (1.0 - bits) * lp0 + bits * lp1
    return out


def _path_seed(S: int, path_index: int, rep: int) -> int:
    return SEED + 10_000_000 * S + 10_000 * path_index + rep


def _evaluate_path(task):
    S, path_index, path_tuple = task
    n = N_DATA
    N = 1 << n
    M = int(math.floor(2 ** (n * R)))
    tau = int(math.floor(2 ** (n * GAMMA)))
    kernel = grandab_kernel_by_rank(N, M, tau)
    bits = bit_matrix(n)
    state_path = np.asarray(path_tuple, dtype=int)
    class_count = count_switch_paths(n, 2, S, initial_state=INITIAL_STATE)
    log2_class_count = math.log2(class_count)
    oracle_err, mix_err, static_err = [], [], []
    max_mismatch, dplus = [], []
    global_max_mismatch = -np.inf
    for rep in range(REPS):
        rng = np.random.default_rng(_path_seed(S, path_index, rep))
        x = rng.choice(np.array([-1.0, 1.0]), size=n)
        y = np.empty(n, dtype=float)
        for i in range(n):
            y[i] = x[i] + sample_ggn(*STATE_THETA[state_path[i]], 1, rng)[0]
        logcand = candidate_logprob_tensor(y, bits)
        lp_oracle = np.sum(
            np.stack([logcand[:, i, state_path[i]] for i in range(n)], axis=1),
            axis=1,
        )
        lp_mix = switching_mixture_log_scores_from_candidates(
            logcand, S, initial_state=INITIAL_STATE
        )
        lp_static = np.sum(logcand[:, :, INITIAL_STATE], axis=1)
        lse = float(logsumexp(lp_oracle))
        po = np.exp(lp_oracle - lse)
        vals = []
        for lp in (lp_oracle, lp_mix, lp_static):
            ranks = ranks_from_order(deterministic_order(lp))
            vals.append(float(np.dot(po, kernel[ranks - 1])))
        oracle_err.append(vals[0]); mix_err.append(vals[1]); static_err.append(vals[2])
        rr_bits = (lp_oracle - lp_mix) / math.log(2.0)
        obs_max = float(np.max(rr_bits))
        if obs_max > log2_class_count + MISMATCH_TOL:
            raise AssertionError(
                f"switching mismatch bound violated: {obs_max} > {log2_class_count}"
            )
        max_mismatch.append(obs_max)
        dplus.append(float(np.dot(po, np.maximum(rr_bits, 0.0))) / n)
        global_max_mismatch = max(global_max_mismatch, obs_max)
    mix_ex = np.asarray(mix_err) - np.asarray(oracle_err)
    static_ex = np.asarray(static_err) - np.asarray(oracle_err)
    mix_mean, mix_ci = mean_ci95(mix_ex)
    static_mean, static_ci = mean_ci95(static_ex)
    oracle_mean, _ = mean_ci95(oracle_err)
    mm_mean, mm_ci = mean_ci95(max_mismatch)
    dp_mean, dp_ci = mean_ci95(dplus)
    return {
        "S": S,
        "path_index": path_index,
        "path": "".join(map(str, state_path.tolist())),
        "exact_path_count": count_exact_switch_paths(n, 2, S, initial_state=INITIAL_STATE),
        "mixture_path_class_count": class_count,
        "log2_path_class_count": log2_class_count,
        "oracle_error_mean": oracle_mean,
        "switching_excess_mean": mix_mean,
        "switching_excess_ci95": mix_ci,
        "initial_state_static_excess_mean": static_mean,
        "initial_state_static_excess_ci95": static_ci,
        "max_log2_mismatch_mean": mm_mean,
        "max_log2_mismatch_ci95": mm_ci,
        "D_plus_bits_per_symbol": dp_mean,
        "D_plus_ci95": dp_ci,
        "max_observed_log2_mismatch": global_max_mismatch,
    }


def run(write: bool = True):
    n = N_DATA
    N = 1 << n
    M = int(math.floor(2 ** (n * R)))
    tau = int(math.floor(2 ** (n * GAMMA)))
    h_states = [conditional_entropy_bpsk_ggn(float(b), float(e)) for b, e in STATE_THETA]
    h_bar = max(h_states)
    if not (h_bar < GAMMA < 1.0 - R):
        raise RuntimeError(f"switching experiment violates theorem region: {h_bar=} {GAMMA=} {1-R=}")

    tasks = []
    for S in S_VALUES:
        paths = list(enumerate_paths(n, 2, S, initial_state=INITIAL_STATE, exact_switches=True))
        expected = count_exact_switch_paths(n, 2, S, initial_state=INITIAL_STATE)
        if len(paths) != expected:
            raise AssertionError("exact path enumeration/count mismatch")
        tasks.extend((S, i, p) for i, p in enumerate(paths))

    workers = MAX_WORKERS
    if workers <= 1:
        path_rows = [_evaluate_path(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            path_rows = list(ex.map(_evaluate_path, tasks))
    path_rows.sort(key=lambda r: (int(r["S"]), int(r["path_index"])))

    summary_rows = []
    for S in S_VALUES:
        rr = [r for r in path_rows if int(r["S"]) == S]
        mix_arr = np.asarray([float(r["switching_excess_mean"]) for r in rr])
        static_arr = np.asarray([float(r["initial_state_static_excess_mean"]) for r in rr])
        oracle_arr = np.asarray([float(r["oracle_error_mean"]) for r in rr])
        summary_rows.append({
            "S": S,
            "exact_path_count": len(rr),
            "mixture_path_class_count": int(rr[0]["mixture_path_class_count"]),
            "log2_path_class_count": float(rr[0]["log2_path_class_count"]),
            "oracle_error_path_average": float(np.mean(oracle_arr)),
            "switching_excess_path_average": float(np.mean(mix_arr)),
            "switching_excess_path_min": float(np.min(mix_arr)),
            "switching_excess_path_max": float(np.max(mix_arr)),
            "initial_state_static_excess_path_average": float(np.mean(static_arr)),
            "initial_state_static_excess_path_min": float(np.min(static_arr)),
            "initial_state_static_excess_path_max": float(np.max(static_arr)),
            "max_observed_log2_mismatch": max(float(r["max_observed_log2_mismatch"]) for r in rr),
        })

    meta = {
        "n": n,
        "reps_per_path": REPS,
        "R": R,
        "gamma": GAMMA,
        "N": N,
        "M": M,
        "tau": tau,
        "initial_state_known": INITIAL_STATE,
        "state0_beta": STATE_THETA[0, 0],
        "state0_eta": STATE_THETA[0, 1],
        "state1_beta": STATE_THETA[1, 0],
        "state1_eta": STATE_THETA[1, 1],
        "h_state0": h_states[0],
        "h_state1": h_states[1],
        "h_bar": h_bar,
    }
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (DATA_DIR / "switching_channel_pathwise.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(path_rows[0].keys()))
            w.writeheader(); w.writerows(path_rows)
        with (DATA_DIR / "switching_channel.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        (DATA_DIR / "switching_channel_meta.txt").write_text(
            "\n".join(
                f"{k}={v:.17g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in meta.items()
            ) + "\n"
        )
    return summary_rows, path_rows, meta


if __name__ == "__main__":
    summary, _, meta = run(True)
    print(meta)
    for r in summary:
        print(r)
