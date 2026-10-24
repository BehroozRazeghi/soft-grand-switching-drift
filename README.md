# Soft GRAND under Channel Switching and Drift

Code and numerical results for **Soft GRAND under Channel Switching and Drift**.

**Paper:** [arXiv:2608.22685](https://arxiv.org/abs/2608.22685)

The paper studies posterior mismatch in soft GRAND when the channel switches within a block or drifts across blocks. The numerical experiments use BPSK with generalized-Gaussian noise and evaluate switching-mixture ordering and periodic pilot refresh.

## Source-code status

The GRAND-based decoding implementation used in the experiments is not distributed in this repository because the applicability of third-party patent rights to this implementation remains unresolved.

The implementation was developed independently and does not copy, modify, translate, or incorporate source code from GRAND-MATLAB, SOGRAND-C, or other existing GRAND codebases.

The experiment scripts are included to record the numerical procedures and parameter settings used in the paper. They depend on source modules that are not distributed here and therefore cannot be executed from this repository alone. The archived numerical data, analysis scripts, and figure-generation script are public.

## Repository structure

```text
analysis/       analysis of archived numerical results
data/           numerical results used in the paper
experiments/    experiment scripts and parameter settings
figures/        manuscript figures
scripts/        figure generation from archived data
src/            source-code availability notice
```

## Environment

Python 3.10 or later is recommended. The public analysis and plotting scripts use NumPy, SciPy, and Matplotlib.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are the versions used to verify this public release.

## Numerical experiments

### Channel switching

The switching experiment uses two generalized-Gaussian channel states,

```text
(beta, sigma) = (1.05, 0.75)
(beta, sigma) = (1.95, 1.30)
```

with

```text
n = 10
S = {0,1,2,3}
R = 0.25
gamma = 0.65
observations per path = 64
```

The initial state is fixed and known. For exact switch counts `S = 0,1,2,3`, the experiment evaluates `1, 9, 36, 84` state paths, respectively. It compares the matched path-conditioned posterior order, the switching-mixture order over paths with at most `S` switches, and the static order based on the initial channel state.

All `2^10` correction patterns are enumerated. GRANDAB error is computed using the exact random-subset finite-population expression; random codebooks are not simulated.

Relevant files:

```text
experiments/exp_switching_channel.py
data/switching_channel.csv
data/switching_channel_pathwise.csv
data/switching_channel_meta.txt
```

### Pilot-estimation calibration

The pilot calibration uses five parameter points, sample sizes

```text
m = {20,25,30,40,50,60,75,100,120,150},
```

and 64 repetitions per setting. The reported calibration constant is the maximum, over the tested settings, of the upper endpoint of the pointwise 95% Student-t interval for

```text
sqrt(m) * E ||theta_hat - theta||.
```

Its numerical value is

```text
C_est = 3.598833911325183.
```

This is an empirical calibration quantity, not a simultaneous confidence bound or a finite-sample proof of a uniform root-`m` risk bound.

Relevant files:

```text
experiments/exp_pilot_mle_calibration.py
data/pilot_mle_calibration.csv
data/pilot_mle_calibration_meta.txt
```

### Refresh-period study

The refresh-period experiment uses

```text
B = 120
n = 10
phi = m/(n q) = 0.5
q = {4,5,6,8,10,12,15,20,24,30}
```

for five deterministic drift periods. Every tested `q` divides `B`.

The empirical grid minimizer of the average tracking error is compared with the continuous minimizer

```text
q* = [C_est / (2 v sqrt(phi n))]^(2/3),
```

where `v = V_B/B`. The quantity `q*` minimizes the tracking upper bound used in the paper; it is not asserted to minimize the finite-block tracking error or decoding error.

Relevant files:

```text
experiments/exp_refresh_scaling.py
data/refresh_scaling.csv
data/refresh_scaling_summary.csv
data/refresh_scaling_raw.csv
```

### Pilot-tracked decoding

The tracked-decoding experiment uses

```text
B = 120
n = 10
T = {120,40}
q = {4,5,6,8,10,12,15,20,24,30}
phi = 0.5
R = 0.25
gamma = 0.65
paired repetitions = 64
```

It compares the matched current-parameter posterior order, the pilot-tracked posterior order, and the static order based on the initial channel parameter. For each sampled observation, all `2^n` correction patterns are enumerated and the exact random-subset GRANDAB error is averaged under the matched correction posterior.

Relevant files:

```text
experiments/exp_pilot_drift.py
data/pilot_drift.csv
data/pilot_drift_meta.txt
```

### Blocklength robustness

The supplementary robustness calculation uses

```text
n = {8,10,12}
```

for representative slow- and fast-drift settings, with eight paired repetitions. It is a finite-block robustness calculation, not an asymptotic convergence study.

Relevant files:

```text
experiments/exp_blocklength_robustness.py
data/blocklength_robustness.csv
```

## Figures

The figures can be regenerated from the archived CSV files without the non-public source implementation:

```bash
python scripts/generate_figures.py
```

The combined manuscript figure is `figures/fig_dynamic_evaluation.pdf`. The four panels are also provided separately.

## Analysis

The following scripts operate only on the archived numerical data:

```bash
python analysis/summarize_switching_paths.py \
  --input data/switching_channel_pathwise.csv

python analysis/bootstrap_refresh_minimizer.py \
  --input data/refresh_scaling_raw.csv
```

## License

The files distributed in this repository are released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).

No rights under third-party patents are granted. See [`PATENT_NOTICE.md`](PATENT_NOTICE.md).

## Citation

```bibtex
@misc{razeghi2026softgrand,
  title         = {Soft GRAND under Channel Switching and Drift},
  author        = {Razeghi, Behrooz},
  year          = {2026},
  eprint        = {2608.22685},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT}
}
```
