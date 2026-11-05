# Results

All numbers below come from the **standard** profile (seed 20260723, 600 simulated channels per
condition per data set), produced by the scripts in this repository. "Measured" values are read
from digitised figures of the paper or quoted from its text/tables; nothing measured was
re-processed. Raw outputs: `results/tables/`, `results/experiments/*/results.json`.

## 1. Comparison table
| # | Quantity | Paper | Ours | Agreement | Classification |
|---|---|---|---|---|---|
| 1 | Truncation map, regime boundaries (Figs. 14-15 setting) | shaded at ~14.5 / 21.7 / 36.2 dB | 14.48 / 21.71 / 36.19 dB (analytic) | exact | Independent numerical reproduction |
| 2 | Inverse mapping, Figs. 14-15 | "excellent agreement" | re-observed = actual spreads to < 1e-6; 100 % convergence, max kappa error 8.7e-10 over 2000 random cases (exp. 3) | yes | Independent numerical reproduction |
| 3 | Discretisation correction | reduces grid error (no numbers) | median spread error on the 1 ns/10 deg grid 12.8/10.0/10.4 % -> 1.6/1.8/1.9 % (delay/DoA/DoD, exp. 3); in the generator 21 % -> 3.7 % (LoS delay, exp. 6) | qualitatively yes | Independent numerical reproduction (table design ours) |
| 4 | Mean cluster number (drawn) | 11.36 LoS / 88.30 NLoS | 12.6 / 86.1 (incl. LoS cluster / rounding) | yes | Statistical reproduction |
| 5 | Mean observable clusters at 35 dB | = measured counts (11.4 / 88.1 digitised) | 7.6 / 60.0 | **no** - see D13 | Statistical reproduction |
| 6 | LD setting (no LoS cluster, 15 dB) mean clusters | 3 / 9 (measured) | 4.1 / 10.9 | close | Approximate reproduction |
| 7 | Per-cluster spreads re-observed on the grid vs Table I | fits in Figs. 6, 10, 11 | median deviation from drawn target 2-4 % | yes | Statistical reproduction |
| 8 | Fig. 17 CvM, delay LoS | 0.004 (recomputed from digitised curves 0.0057) | 0.0155 (sampling null 95 %: 0.023) | consistent with measured | Statistical reproduction |
| 9 | Fig. 17 CvM, DoA LoS | 0.007 (0.0089) | 0.0347 (null 0.023); median 0.19 vs 0.25 | deviates (too narrow) | Statistical reproduction |
| 10 | Fig. 17 CvM, DoD LoS | 0.003 (0.0043) | 0.0053 (null 0.022) | consistent | Statistical reproduction |
| 11 | Fig. 17 CvM, delay NLoS | 0.007 (0.0046) | 0.0310 (null 0.025); median 22.1 m vs 17.5 m | deviates | Statistical reproduction |
| 12 | Fig. 17 CvM, DoA NLoS | 0.020 (0.0153) | 0.0897 (null 0.029); median 0.89 vs 0.57 | deviates | Statistical reproduction |
| 13 | Fig. 17 CvM, DoD NLoS | 0.029 (0.0228) | 0.1374 (null 0.028); median 0.89 vs 0.54 | deviates | Statistical reproduction |
| 14 | Table II alpha-beta LoS omni (alpha / beta / sigma) | 76.37 / 1.80 / 1.98 (synthetic), [21] 76.86 / 1.77 / 1.28 | median 72.19 / 1.96 / 1.76; 95 % range [70.5, 74.6] / [1.82, 2.06] / [1.39, 2.11] | CIs overlap in 80 / 99.5 / 97.5 % of repeats | Statistical reproduction |
| 15 | Table II CI model LoS omni n | 1.86 [1.83, 1.90] | 1.75 [1.72, 1.78] | **no overlap** | Statistical reproduction |
| 16 | Table II alpha-beta NLoS omni | 96.22 / 1.56 / 6.98 | 105.1 / 1.03 / 5.73 | CIs overlap in 99 / 99 / 94 % | Statistical reproduction |
| 17 | Table II CI model NLoS omni n / sigma | 2.92 / 8.05 | 2.75 / 8.93 | overlap 98 / 97 % | Statistical reproduction |
| 18 | Truncated MLE vs OLS (Figs. 12-13 method), NLoS slope | MLE steeper than OLS | truth -1.81, MLE -1.67, OLS -1.60 | yes (direction) | Independent numerical reproduction |
| 19 | Same, LoS slope | MLE steeper than OLS | truth -1.19, MLE -0.50, OLS -0.54 | **no** | Independent numerical reproduction |
| 20 | NCP coverage +-1 sigma / +-3 sigma | LoS 71.55 / 99.16 %, NLoS 67.79 / 99.34 % (measured) | LoS 73.1 / 99.35 %, NLoS 72.3 / 99.76 % (simulated, own MLE) | close | Statistical reproduction |
| 21 | ML-KPD vs KPD | ML-KPD separates sub-structures | ARI 0.85 vs 0.78 (LoS), 0.87 vs 0.45 (NLoS) on generated ADPSs (exp. 5) | yes | Approximate reproduction |
| 22 | Table I extraction, Fig. 1, measured histograms | - | - | - | Not reproducible (data unavailable) |

Table II, max-dir columns and all CI values: `results/tables/pathloss_table2.csv`.
Full CvM table: `results/tables/cvm_validation.csv`.

## 2. Reading the results
* **The mathematics of Sec. VI is reproduced exactly.** The truncation map, its regimes,
  the Newton inverse and the role of the discretisation correction behave as described.
* **LoS channels agree with the measurements** for delay and DoD spread within the
  sampling uncertainty of 21 measured links; DoA spread is somewhat narrower than measured
  (and than the authors' synthetic channels).
* **NLoS channels are clearly more dispersive** than both the measurements and the authors'
  generator. Experiment 2 shows that NLoS dispersion grows steeply with the number of
  observable clusters (2.9 -> 60 clusters from 5 to 35 dB). The drawn Table I statistics
  (88 clusters, uniform centroids, NCP sigma ~ 10 dB) leave many off-boresight clusters
  within 35 dB. Experiment 7 shows that the ambiguous parameters (Table I vs text arrival
  times, normalised vs wrapped Laplace) change the CvM values only marginally; they do not
  explain the gap. With the published Table I the gap stays; the labelled calibrated set of
  Section 5 closes it (LIMITATIONS.md items 9, 18).
* **Path loss** is consistent with Table II in most parameters within the (wide) 50-channel
  confidence intervals; the LoS level is ~3 dB lower than [21] over 1-100 m, so the narrow
  CI-model exponent interval does not overlap.
* **Truncation-aware power fitting** helps where truncation acts like a threshold on NCP
  (NLoS: half the OLS bias removed). In the generator, observability depends on the cluster's
  peak tap, i.e. also on its shape, so the single-level truncated likelihood is only an
  approximation; in LoS (few, strong clusters near the cut-off) both estimators are biased.
  With an exact NCP threshold the estimator recovers the truth (tests/test_cluster_model.py).

## 3. Performance (exp. 1, one process, 4-core laptop)
0.72 s per LoS and 2.19 s per NLoS realization including the metrics; data-set generation
with 4 workers: 600 channels in 7 min (LoS) / 9-20 min (NLoS). The cut-off fixed point
needed at most 3 sweeps in all 2400 standard-profile channels.

## 4. Figures
`results/figures/FIGURES.md` lists every figure with its content, data source and
classification. Figures 2, 6, 10, 11 and 17 overlay digitised published curves; Figures 3-5,
7-9 and 12-13 show the published model and our simulated draws only; Figures 14-16 are
reproduced independently.

## 5. GNU Radio implementation (results/gnuradio/)
The channel model runs inside two GNU Radio 3.10 flowgraphs (`gnuradio/`, see gnuradio/README.md).
`scripts/run_gnuradio.py` sounded 1960 channels through the no-GUI sounder flowgraph (VNA sweep ->
noise -> IFFT -> |.|^2/N_f -> ADPS -> 35 dB taps): per parameter set 210 LoS / 170 NLoS at the measured
link distances and 300 per condition at d ~ U(1, 100) m.

**Sounder chain verification:** median |GNU Radio - generator| <= 0.36 % for the spreads and <= 0.004 dB
for the path gains (`gr_sounder_verification.csv`).

**Fig. 17 (omega^2 vs measured; sampling null 95 % about 0.023 LoS / 0.03 NLoS):**

| statistic | paper (published) | GNU Radio, Table I | GNU Radio, calibrated |
|---|---|---|---|
| LoS delay | 0.004 | 0.0138 | 0.0037 |
| LoS DoA | 0.007 | 0.0337 | 0.0132 |
| LoS DoD | 0.003 | 0.0029 | 0.0110 |
| NLoS delay | 0.007 | 0.0265 | 0.0070 |
| NLoS DoA | 0.020 | 0.0756 | 0.0041 |
| NLoS DoD | 0.029 | 0.1176 | 0.0121 |

Table I: 3 of 6 within the null; calibrated: 6 of 6.

**Table II (omni, median of 50-channel fits; paper synthetic in brackets):** calibrated LoS alpha 74.2
(76.37), beta 1.96 (1.80), n 1.87 (1.86); NLoS alpha 105.1 (96.22), beta 1.16 (1.56), n 2.88 (2.92).
The 95 % CIs overlap the paper's in >= 71 % of repeats for every parameter (`gr_pathloss.csv`). With
Table I the LoS close-in exponent was 1.74 with no CI overlap.

