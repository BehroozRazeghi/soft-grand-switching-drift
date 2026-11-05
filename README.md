# Soft GRAND under Channel Switching and Drift

Numerical results and supporting scripts for:

**Behrooz Razeghi, “Soft GRAND under Channel Switching and Drift,” *IEEE Communications Letters*, 2026.** 
[IEEE Xplore](https://ieeexplore.ieee.org/document/11771020) · [DOI](https://doi.org/10.1109/LCOMM.2026.3749996) · [arXiv](https://arxiv.org/abs/2608.22685)

This repository contains the archived numerical results, experiment specifications, data-analysis scripts, and figure-generation code. The GRAND-based decoding modules used to produce the numerical data are **not distributed** because the applicability of third-party patent rights remains unresolved. Consequently, the original experiments cannot be rerun from this repository alone. The archived results can, however, be analyzed and the manuscript figures regenerated using the public scripts.

## Research scope

Soft GRAND orders candidate correction patterns according to a posterior distribution. Channel switching within a block, or parameter drift between blocks, can make the ordering distribution differ from the posterior associated with the channel that generated the observation.

The paper examines three aspects of this mismatch.

### Finite-block reliability

For matched posterior $p$, query-ordering posterior $Q$, and true correction $E$ given observation $r$, the query rank satisfies

$$
\log_2 G_Q(E\mid r)\leq -\log_2 p(E\mid r)
+\left[\log_2\frac{p(E\mid r)}{Q(E\mid r)}\right]^+.
$$

The rank bound is combined with finite-population collision probabilities to analyze GRAND with abandonment (GRANDAB) for a random-subset code ensemble.

### Within-block switching

A uniform mixture of the posteriors indexed by admissible channel-state paths has log-posterior mismatch at most the logarithm of the path-class size. For a fixed finite collection of memoryless channels, the paper establishes uniform reliability below the minimum constituent capacity when uniform input is capacity-achieving for every state and the maximum switch count satisfies $S_n\log n=o(n)$.

### Block-to-block drift

With periodic estimation from $m$ pilots every $q$ data blocks, the average parameter-tracking error is bounded by $\varepsilon_m+qV_B/B$, where $\varepsilon_m$ bounds the expected estimation error and $V_B$ is total parameter variation. Under an estimation-risk bound of order $m^{-1/2}$, this gives a continuous minimizer of the *tracking upper bound*, not of the actual finite-block tracking or decoding error.

The numerical study uses BPSK transmission with independent generalized-Gaussian noise samples, whose parameters may switch within a block or drift between blocks. The estimation parameter is $\vartheta=(\beta,\eta)$, with $\eta=\ln\sigma$.

## Repository contents

| Path | Contents |
| --- | --- |
| `data/` | Archived numerical results and experiment metadata |
| `experiments/` | Experiment specifications and scripts used to generate the archived data; require undistributed `src` modules |
| `analysis/` | Analyses that operate on the archived CSV files |
| `scripts/generate_figures.py` | Regenerates the four figure panels and combined figure from archived data |
| `figures/` | Manuscript figures |
| `src/` | Source-code availability notice |
| `requirements.txt` | Pinned dependencies for the publicly executable scripts |
| `PATENT_NOTICE.md` | Scope of the source-code and patent notices |

## Numerical methodology

The main decoding experiments use blocklength $n=10$, nominal rate parameter $R=0.25$, and abandonment exponent $\gamma=0.65$. With the definitions used in the paper, these give an ambient space of $2^{10}=1024$ words, random-subset codebook size $M=\lfloor 2^{nR}\rfloor=5$, and query budget $\tau=\lfloor 2^{n\gamma}\rfloor=90$.

All $2^n$ candidate corrections are enumerated for each sampled channel observation. Conditional on that observation, the GRANDAB error is calculated by averaging the **exact random-subset collision expression** over the matched correction posterior. Random codebooks are not simulated. The remaining averages over channel observations and, where applicable, pilot estimates are estimated by Monte Carlo. These quantities are ensemble-average error probabilities, **not measured block-error rates for a specified code**.

### 1. Within-block channel switching

The two channel states have generalized-Gaussian parameters

| State | Shape $\beta$ | Scale $\sigma$ |
| --- | ---: | ---: |
| 1 | 1.05 | 0.75 |
| 2 | 1.95 | 1.30 |

The initial state is known and fixed. For each exact number of switches $S\in\lbrace0,1,2,3\rbrace$, the experiment evaluates every state path starting in that state: respectively **1, 9, 36, and 84 paths**. Each path is evaluated using 64 sampled channel observations.

Three posterior orderings are compared: the matched path-conditioned ordering (oracle), the uniform mixture over paths with **at most** $S$ switches, and a static ordering that retains the initial channel state. The plotted quantity is the excess ensemble-average GRANDAB error relative to the oracle, summarized across paths with exactly $S$ switches.

For $S=3$, the mean estimated excess errors across the 84 paths are $2.586\times10^{-3}$ (switching mixture) and $4.938\times10^{-3}$ (static ordering). The mixture has a smaller estimated excess error on 81 of those paths. These are finite-block numerical results for the specified channel states, not a general performance guarantee.

Files: `experiments/exp_switching_channel.py`, `data/switching_channel.csv`, `data/switching_channel_pathwise.csv`, and `data/switching_channel_meta.txt`.

### 2. Pilot-estimator calibration

The constrained maximum-likelihood estimator is evaluated at five parameter points and ten pilot sample sizes,

$$
m\in\lbrace20,25,30,40,50,60,75,100,120,150\rbrace,
$$

with 64 independent repetitions per setting. The reported calibration value,

$$
\widehat C_{\mathrm{cal}}=3.598833911325183,
$$

is the maximum of the upper endpoints of 50 **pointwise** 95% Student-$t$ intervals for $\mathbb{E}[\sqrt m\,\|\widehat\vartheta_m-\vartheta\|]$.

This finite-grid quantity is used as a numerical surrogate in the refresh-period comparison. It is **not** a simultaneous confidence bound and does not establish the uniform, eventual estimation-risk bound of order $m^{-1/2}$ assumed in the theoretical result.

Files: `experiments/exp_pilot_mle_calibration.py`, `data/pilot_mle_calibration.csv`, and `data/pilot_mle_calibration_meta.txt`.

### 3. Refresh period and parameter tracking

The tracking study uses $B=120$ data blocks, $n=10$, and deterministic parameter paths with periods $T\in\lbrace120,80,60,40,30\rbrace$. The tested refresh intervals are

$$
q\in\lbrace4,5,6,8,10,12,15,20,24,30\rbrace.
$$

Every $q$ divides $B$. The pilot-to-data ratio is fixed at $\phi=m/(nq)=0.5$, so $m=5q$ pilots are collected at each refresh, corresponding to one pilot symbol per two data symbols. Each setting uses 64 repetitions. The empirical grid minimizer is assessed with 5,000 bootstrap resamples.

Under the assumed estimation-risk bound of order $m^{-1/2}$, the continuous tracking-bound surrogate and its minimizer, for $v=V_B/B>0$, are

$$
f(q)=\frac{C_{\mathrm{est}}}{\sqrt{\phi nq}}+qv,
\qquad
q^\star=\left(\frac{C_{\mathrm{est}}}{2v\sqrt{\phi n}}\right)^{2/3}.
$$

The numerical comparison substitutes $\widehat C_{\mathrm{cal}}$ for $C_{\mathrm{est}}$. The resulting $q^\star$ minimizes this **calibrated surrogate**, whereas the empirical grid minimizer minimizes the sampled mean tracking error over the tested values of $q$. The two need not coincide.

Files: `experiments/exp_refresh_scaling.py`, `data/refresh_scaling.csv`, `data/refresh_scaling_summary.csv`, and `data/refresh_scaling_raw.csv`.

### 4. Pilot-tracked decoding

For the drift periods $T\in\lbrace120,40\rbrace$, the experiment compares the matched current-parameter ordering, the periodically updated pilot-based ordering, and a static ordering based on the initial parameter. It uses the same $B$, $n$, $q$ grid, $\phi$, $R$, and $\gamma$ as above, with 64 paired repetitions. Within each repetition, the policies are evaluated on the same sampled data observations; pilot samples differ across refresh intervals.

The reported decoding quantity is excess ensemble-average GRANDAB error relative to matched ordering. The tracking-bound minimizer is shown for comparison, but is **not** claimed to minimize decoding error. The manuscript reports empirical decoding-error minima at $q=20$ for $T=120$ and $q=8$ for $T=40$ on the tested grid.

Files: `experiments/exp_pilot_drift.py`, `data/pilot_drift.csv`, and `data/pilot_drift_meta.txt`.

### 5. Blocklength check

A supplementary calculation considers $n\in\lbrace8,10,12\rbrace$ for two selected period–refresh pairs, $(T,q)=(120,20)$ and $(40,8)$, using eight repetitions. This is a small finite-block comparison and does not constitute an empirical asymptotic study.

Files: `experiments/exp_blocklength_robustness.py` and `data/blocklength_robustness.csv`.

## Reproduce figures and analyze archived results

The public scripts require **Python 3.11–3.14**, consistent with the pinned NumPy and SciPy versions in `requirements.txt`. From a local checkout:

```bash
git clone https://github.com/BehroozRazeghi/soft-grand-switching-drift.git
cd soft-grand-switching-drift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Regenerate the combined manuscript figure and four individual panels:

```bash
python scripts/generate_figures.py
```

The script writes PDF and PNG files to `figures/`, including `figures/fig_dynamic_evaluation.pdf` and the corresponding standalone panels. It reads the archived CSV files and does not require the undistributed decoding modules.

Two additional analyses can be run directly:

```bash
python analysis/summarize_switching_paths.py \
  --input data/switching_channel_pathwise.csv

python analysis/bootstrap_refresh_minimizer.py \
  --input data/refresh_scaling_raw.csv
```

These commands write `data/switching_path_summary_extended.csv` and `data/refresh_minimizer_bootstrap.csv`, respectively. The latter estimates the bootstrap selection frequency of each tested refresh interval; it does not produce new decoding simulations.

**Re-running the original experiments:** Scripts under `experiments/` import modules from `src/` that are not present in this release. Installing `requirements.txt` does not make those scripts executable. To inspect the experimental settings and the stored outputs, use the scripts and metadata files listed above.

## Source code and licensing

The GRAND-based decoding implementation is withheld because the applicability of third-party patent rights has not been resolved. The implementation was independently developed and does not incorporate code from GRAND-MATLAB, SOGRAND-C, or other GRAND implementations. See [`PATENT_NOTICE.md`](PATENT_NOTICE.md) for the notice associated with this release.

The distributed files are licensed under the [BSD 3-Clause License](LICENSE). That copyright license does not confer rights under third-party patents.

## Citation

Please cite the journal article when referring to the research:

```bibtex
@ARTICLE{11771020,
  author  = {Razeghi, Behrooz},
  journal = {IEEE Communications Letters},
  title   = {Soft GRAND under Channel Switching and Drift},
  year    = {2026},
  doi     = {10.1109/LCOMM.2026.3749996}
}
```
