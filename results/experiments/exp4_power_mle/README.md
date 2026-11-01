# Experiment 4 - cluster power model: truncated MLE vs OLS

**Purpose.** Re-estimate (alpha, beta, rho, eta) from simulated OBSERVABLE clusters, where the truth (Table I) is known.

**Configuration.** see `config.json` (profile `standard`, seed `20260723`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- LOS (2612 observable of 4547 drawn): truth beta -1.19; MLE beta -0.50 (alpha -18.26, rho 7.43, eta -0.244); OLS beta -0.54 (alpha -17.65); L-BFGS LLF -3046.1 vs ours -3024.6
- LOS coverage +-1 sigma: 0.731 (truncated expectation 0.701); +-3 sigma 0.9935
- NLOS (23769 observable of 34121 drawn): truth beta -1.81; MLE beta -1.67 (alpha -31.36, rho 9.52, eta -0.045); OLS beta -1.60 (alpha -30.62); L-BFGS LLF -41377.2 vs ours -41122.7
- NLOS coverage +-1 sigma: 0.723 (truncated expectation 0.716); +-3 sigma 0.9976

## Comparison with the paper
- paper (Figs. 12-13): OLS slope is less steep than the truncation-aware MLE - reproduced qualitatively if OLS beta > MLE beta here
- paper coverage (measured): LoS 71.55% / 99.16%, NLoS 67.79% / 99.34% (expected 74.6/99.8 and 71.7/99.8)

## Limitations
- truncation in the generator is a peak-tap criterion, not exactly 'NCP below a level', so the truncated-normal likelihood is only approximately correct here; residual bias is therefore expected
- learning rate 1e-3 with block alternation; iteration budget reduced outside the 'full' profile
