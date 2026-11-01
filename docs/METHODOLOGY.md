# Methodology and implementation decisions

Every place where the paper is ambiguous or silent, and what we did. Code references are
to `src/channel_model/`.

## 1. Grid and units (`adps.py`)
Delay bins 0.29979 m (1 ns), 1001 bins (300 m); angles phi_k = (k - 18) * 10 deg, k = 0..35,
so boresight (0) is on the grid. Delays in metres, angles in radians, power in linear
units internally and dB at interfaces. Centroids are snapped to the nearest grid bin.

## 2. ADPS from frequency data (Eq. 1, `adps_from_frequency_response`)
`P = |IFFT_{N_f N_os}(H)|^2 / N_f`, evaluated on an N_os-times finer delay grid; each
critical bin keeps the maximum of the N_os fine points centred on it (scalloping
compensation without inventing resolution). N_os = 5 (paper: unspecified). With N_os = 1
the result satisfies Parseval. Optional Hann window (off by default; only for the
clustering sensitivity test).

## 3. MPC taps (`mpc_extraction.py`)
All ADPS entries >= max - 35 dB. A synthetic sounder (`synthetic_frequency_response`,
Gaussian 13 deg beams on the 10 deg rotor grid) exists **only as test data**.

## 4. KPD / ML-KPD (`kpd_clustering.py`)
* Features: tau / s_tau and, per angle, (cos phi, sin phi) / circular std - wrap-around aware.
* Density: rho_i = sum_{j in kNN(i)} exp(P_j / P_max) exp(-d_ij^2) (power weighting as in He et al.).
* Relative density rho_i / max over the neighbourhood; key points = 1; parent = nearest
  higher-density neighbour; initial clusters = roots.
* Merging: two initial clusters merge if some kNN edge between them has
  min(relative density) >= threshold; connected components via sparse graph.
* Level-1 threshold 0.5 (paper: "conservative", value not given). On generated
  channels ARI is unchanged for thresholds 0-0.5 (tested).
* Separability: projections on delay/DoA/DoD, moving average 5/3/3 bins (circular for
  angles, cut at the minimum), peaks merged if spacing <= window or dip >= 0.8 x lower peak.
* Re-clustering: K' = sqrt(M_c/2), threshold from 1 down by 0.01 until #sub-clusters <= #modes;
  recursion until no cluster splits (max 6 levels).
K-means/DBSCAN are not used.

## 5. Distributions (`distributions.py`, `cluster_model.py`)
Gamma(shape, scale); counts rounded with floor(x + 0.5). LoS: 1 LoS cluster + round(Gamma).
NLoS: 1 + round(Gamma). Log-normal in natural log. Bimodal Laplace (Eq. 3) normalised on
[-pi, pi) ("truncated", default) or wrapped. Angular spreads are capped at 0.999.
NCP sigma floor 0.1 dB (rho + eta x can become negative for large x with negative eta).

## 6. Angular spread (Eq. 4)
Fleury: S = sqrt(1 - |sum P e^{j phi}|^2 / (sum P)^2). See PAPER_ANALYSIS D1.

## 7. Truncated cluster statistics (`dynamic_range.py`)
For kappa = (kappa_tau, kappa_R, kappa_T) and L = DR_eff ln10/10 the surviving region is
tau/kappa_tau + |phiR|/kappa_R + |phiT|/kappa_T <= L, |phi| <= pi. Delay moments and the
DoA/DoD resultant lengths are written as nested integrals; the two innermost are closed
form (lower incomplete gamma sums, `_G`), the outer one is piecewise 64-point
Gauss-Legendre with breakpoints at the angular saturation points pi/kappa. Angular spreads
do not depend on kappa_tau and S_tau = kappa_tau g(L, kappa_R, kappa_T). Verified against
brute-force integration (tests: < 1 %) and the DR -> infinity limits.
Discrete version: exact sums on the measurement grid (closed-form geometric sums in delay).
Regime boundaries: L = pi/kappa_wide, pi/kappa_narrow, pi/kappa_wide + pi/kappa_narrow,
giving 14.48 / 21.71 / 36.19 dB for the Fig. 14/15 setting (matches the figure shading).

## 8. Inverse mapping (`inverse_mapping.py`)
Because angular spreads do not depend on kappa_tau, the 3-D problem is solved as a 2-D
Newton iteration on (log kappa_R, log kappa_T) with a complex-step Jacobian (analytic to
machine precision), backtracking line search and a bisection initialisation per angle,
followed by the closed form kappa_tau = S_tau / g. Round-trip error ~1e-10 and 100 %
convergence over kappa_tau in [0.1, 50] m, kappa in [0.02, 3] rad, DR in [1, 50] dB
(experiment 3). A direct 3-D Newton (`invert_spreads_full3d`) is kept for reference; it
diverged from poor starts, which is why the reduced form is used.

## 9. Discretisation correction (`DiscretizationCorrection`)
Table Delta(DR, kappa_tau, kappa_R, kappa_T) = discrete-grid spreads - continuous
spreads on a 22 x 26 x 24 x 24 grid (DR 0.1-60 dB, kappa_tau 0.05-400 bins, kappa 0.005-30 rad,
log-spaced), multilinear interpolation, built once (~2 min) and cached. Usage
(`solve_decay_constants` and the generator): invert S_obs, then 3 times: Delta at the
current kappa, invert S_obs - Delta. Delta is bounded to 50 % of the target spread:
for clusters only a few dB above the cut-off with very narrow angular decay the
discrete delay spread can exceed the continuous one by tens of bins (the grid keeps only
the centre angular bin, which then uses the whole delay budget), and an unbounded
correction made the refinement run away (kappa_tau of thousands of metres). This bound
is our addition.

## 10. Generator (`channel_generator.py`, Fig. 16)
1-3. Draws as in section 5; cluster power [dB] = NCP - FSPL(d_LoS) with d/d_LoS =
(d_LoS + arrival)/d_LoS.
4-6 coupling. The drawn power is the *observable* cluster power (sum of surviving taps,
Eq. 5), so the peak tap depends on the shape, the shape on DR_eff = peak - cut-off, and the
cut-off on all peaks. We solve this fixed point: peak := power, cut-off := max peak - DR;
iterate (correction + inversion at the current DR_eff, peak = power - 10log10(sum of
surviving normalised taps)) with a damped cut-off update from the third sweep on. It
converges in 2-3 sweeps. A final pass re-solves all clusters at the final cut-off.
Clusters whose peak is not above the cut-off are not observable and are dropped.
6. Each observable cluster contributes only its own above-cut-off taps (exactly the
truncated shape the inversion assumed); 7. sum and global threshold.
Clusters arriving beyond the 300 m window are dropped (counted).

## 11. Channel metrics (`channel_metrics.py`, following [21])
Omni PDP = max over orientation pairs per delay bin; max-dir PDP = orientation pair with
the largest total power; PL = -10 log10 sum PDP; RMS delay spread of the omni PDP;
DoA/DoD APS = ADPS summed over delay and the other angle; Fleury spreads.

## 12. Power-model fitting (`power_model.py`)
* Eq. 10 weights over 20 bins of x = 10log10(d/d_LoS) in [0, 20] dB.
* LLF: sum w [-ln sigma - (y - mu)^2 / (2 sigma^2) - ln Phi((mu - thr)/sigma)] with analytic
  gradient (inverse Mills ratio; checked against finite differences).
* Optimiser: fixed learning rate 1e-3, alternating (alpha, beta) / (rho, eta) blocks; a
  step that would lower the LLF or make sigma <= 0 is halved (safeguard, our addition).
  Multi-start: start 0 from OLS, the others uniform in a box. L-BFGS-B is run only as a check.
* Threshold for synthetic data: per realization, NCP_thr = cut-off + FSPL(d_LoS) (a
  cluster with total power below one tap at the cut-off cannot be seen). The paper's
  "thresholding level" for measured data is not defined further.

## 13. Validation statistics (`validation.py`)
CvM distance omega^2 = mean over the pooled sample of (F_a - F_b)^2 (normalised; T =
NM/(N+M) omega^2 equals scipy's statistic). Recomputing the paper's own values from the
digitised curves reproduces their magnitude, supporting this definition. A bootstrap null
(two samples of the same law with the same sizes) gives the omega^2 explained by finite
samples alone; we label a result "consistent" if it lies below the null's 95 % quantile.
Validation channels are generated at the 21/17 measured link distances (cycled) - the
paper does not state the distances it used.

## 14. Path loss (`path_loss.py`)
alpha-beta: OLS on 10log10 d with Student-t CIs; sigma CI from chi-square with n-2 dof.
CI model: PL - FSPL(1 m) = 10 n log10 d, slope through the origin, n-1 dof. These choices
reproduce the widths of the Table II intervals (e.g. sigma = 1.98, N = 50 -> [1.66, 2.47]).
Experiment: 50 channels at d ~ U(1, 100) m; to obtain a sampling distribution, 50
channels are drawn without replacement from a pool of n_realizations generated channels,
repeated n_pathloss_repeats times.

## 15. Reproducibility
Every realization has its own random stream `SeedSequence(seed, spawn_key=(condition, data set, index))`,
so results do not depend on the number of worker processes.
