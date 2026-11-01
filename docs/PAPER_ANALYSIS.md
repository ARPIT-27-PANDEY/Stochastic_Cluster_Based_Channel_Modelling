# Paper analysis

**Paper.** Y. Cheng, N. A. Abbasi, J. Gomez-Ponce, Z. Zhang, A. F. Molisch, "Stochastic
Cluster-Based Channel Modeling for Sub-Terahertz Outdoor Device-to-Device Radio
Propagation at 145 GHz", IEEE TWC 25 (2026) 20883-20899, doi 10.1109/TWC.2026.3714332.

Repository contributors: 23116075 - Piyush Bhaskar; 23116005 - Akersheet Tripathi;
23116017 - Arpit Kumar Pandey.

## 1. Problem and contribution
A stochastic, cluster-based double-directional channel model for outdoor D2D links at
145 GHz (both ends at 1.5 m), extracted from measurements, with three claimed contributions:
(i) a multi-level KPD clustering (ML-KPD) of measured ADPSs; (ii) cluster statistics
(number, arrival, centroids, intra-cluster spreads, power) with truncation-aware power
fitting; (iii) a generator that accounts for the measurement's finite dynamic range (DR)
and grid discretisation via an inverse mapping from observable spreads to decay constants.

## 2. Measurement system (Sec. II, [21])
VNA sounder, 145-146 GHz (1 GHz, 1001 points -> 1 ns = 0.29979 m delay bins, 1 us = 300 m
excess-delay window); rotating horns, 13 deg HPBW, 10 deg steps -> 36 x 36 orientation
pairs; 35 dB uniform DR threshold; 21 LoS + 17 NLoS locations, 1-100 m. Delays are
reported in metres, angles in radians, angular spreads (Fleury) are unitless.

## 3. ADPS (Eq. 1)
P(tau, phiR, phiT) = |sum_f H(f, phiR, phiT) e^{j2pi f tau}|^2 per orientation pair. The
typeset equation places |.|^2 inside the sum (typo - it would remove the delay
dependence); we use |sum|^2 / N_f. Delay is oversampled to reduce scalloping (factor not
given; we use 5 and keep the per-bin maximum).

## 4. MPC taps (Sec. IV-A)
All ADPS samples within 35 dB of the global maximum are "MPC taps".

## 5. ML-KPD clustering (Sec. IV-B, Algorithm 1)
KPD [He et al. 2017]: kNN kernel density weighted by power, relative density, key points,
initial clusters by following higher-density neighbours, merge via a connectivity
threshold. Level 1 with K = sqrt(M/2) and a "conservative" threshold; then every
cluster whose delay/DoA/DoD projection is multimodal (moving average 5/3/3 bins, dip ratio
0.8) is re-clustered with K' = sqrt(M_c/2), threshold searched from 1 down in 0.01 steps
until at least half of the modes are kept and the count does not exceed the modes.
Not specified: feature normalisation, kernel, level-1 threshold value, exact merge rule.

## 6. Statistical model (Sec. V, Table I)
| Quantity | LoS cluster | LoS (other) | NLoS |
|---|---|---|---|
| number | 1 | additional ~ Gamma(3.39, 3.35) (mean 11.36 incl. LoS) | N-1 ~ Gamma(1.89, 46.72) |
| arrival [m] | 0 | Gamma(0.86, 32.40) (text: 0.83, 33.36) | Gamma(1.51, 35.00) (text: 1.46, 36.31) |
| delay spread [m] | LogN(-1.42, 0.43) | LogN(-1.42, 0.52) | LogN(-0.79, 0.89) |
| DoA centroid | 0 | bimodal Laplace w=(0.734, 0.266), mu=(0, pi), lambda=(0.292, 0.068) | uniform |
| DoD centroid | 0 | w=(0.318, 0.682), lambda=(0.151, 1.988) | uniform |
| DoA spread | LogN(-1.98, 0.042) | LogN(-2.30, 0.29) | LogN(-2.18, 0.35) |
| DoD spread | LogN(-1.96, 0.048) | LogN(-2.26, 0.32) | LogN(-2.17, 0.38) |
| NCP alpha/beta/rho/eta | 7.63/0/1.85/0 | -20.69/-1.19/9.26/-0.17 | -33.41/-1.81/10.39/-0.035 |

LD setting (LoS cluster removed, 15 dB DR): mean cluster numbers 3 (LoS) / 9 (NLoS).

## 7. Cluster power (Eqs. 5-11)
P_c = sum of the cluster's taps (Eq. 5); NCP = P_c + FSPL(d_LoS) (Eq. 6, also in NLoS);
NCP ~ N(alpha + 10 beta log10(d/d_LoS), (rho + 10 eta log10(d/d_LoS))^2) (Eqs. 7-9).
Fitting: distance-equalisation weights over 20 bins (Eq. 10), truncated-normal weighted
LLF (Eq. 11), alternating gradient ascent with learning rate 1e-3, multi-start.

## 8. Cluster shape (Eqs. 12-13)
Separable exponential (delay) x Laplacian (DoA) x Laplacian (DoD) with decay constants
kappa = (kappa_tau, kappa_R, kappa_T); ADPS = reshape(APS_T kron APS_R kron PDP).

## 9. Dynamic range and inverse mapping (Sec. VI)
With a cut-off DR_eff below a cluster's peak only taps with tau/kappa_tau + |phiR|/kappa_R +
|phiT|/kappa_T <= DR_eff ln10/10 survive; observable spreads S_obs = F(kappa; DR_eff)
(delay-only special case Eq. 14). F is inverted by Newton-Raphson. Figs. 14-15 validate
it (kappa_tau = 10 m, DoD 0.2 pi, DoA 0.3 pi, DR 1-50 dB) and show four regimes.

## 10. Discretisation correction
A pre-computed correction (discrete-grid spreads minus continuous spreads) is subtracted
from the target spreads before the inversion. Its construction is not specified.

## 11. Generation procedure (Fig. 16)
Inputs -> cluster number -> per-cluster parameters -> global cut-off from the strongest
peak (35 dB) -> correction + inverse mapping -> cluster ADPS scaled to its power at its
centroid -> superposition. Validation statistics are extracted from the generated ADPSs.

## 12. Validation (Sec. VII)
Fig. 17: CDFs of channel delay/DoA/DoD spreads, measured vs synthetic; CvM distances
0.004/0.007 (delay LoS/NLoS), 0.007/0.020 (DoA), 0.003/0.029 (DoD).

## 13. Path loss (Sec. VIII, Table II)
50 synthetic channels per condition, distances up to 100 m; alpha-beta and close-in
(CI) models for omni and max-direction PL, compared with [21] (values in the YAML).

## 14. Discrepancies and open points found while reproducing
| # | Item | Paper | Our decision |
|---|---|---|---|
| D1 | Eq. 4 angular spread | typeset without sqrt and with a unit-phasor reference | Fleury sqrt(1-|mu|^2) (consistent with "max 1", Fig. 15 limits and [21] Eqs. 9-10); typeset form kept as `eq4_as_typeset` |
| D2 | Eq. 1 | |.|^2 inside the sum | |sum|^2 |
| D3 | Eq. 10 minimum bin share | 1 % in the equation, 2 % in the sentence above | 1 % default, 2 % configurable |
| D4 | Eq. 11 | written in the path-loss domain | power-domain form (algebraically identical) |
| D5 | arrival Gamma | Table I 0.86/32.40, 1.51/35.00; text 0.83/33.36, 1.46/36.31 | Table I default, text = experiment 7 |
| D6 | Figs. 14/15 | text: DoD 0.2 pi, DoA 0.3 pi; legend/curves suggest the reverse | text assignment, noted on the figure |
| D7 | Fig. 2 NLoS fit | Gamma(1.89, 46.72) stated for N-1, but the plotted fit matches it applied to N (dev 0.0013 vs 0.0094) and the "mean 88.30" equals the Gamma mean | N = 1 + Gamma (as stated); <1 cluster difference |
| D8 | "angular centroids all bimodal Laplace" | NLoS centroids are uniform in Table I | Table I |
| D9 | Eq. 3 normalisation | Laplace pdf on [-pi, pi) without stating wrap or renormalisation | normalised on [-pi, pi) (default) / wrapped (experiment 7) |
| D10 | LoS 0.9 m spacing | text 0.9 m, [21] lists 1 m | [21] distances |
| D11 | Section numbering | intro says validation in Sec. VII | cosmetic |
| D12 | Fig. 14/15 "expected" lines | dashed values ~0.70/0.85 = untruncated wrapped-Laplace spread sqrt(1-1/(1+kappa^2)^2) | both the wrapped limit and the [-pi,pi) truncated limit are drawn |
| D13 | cluster count vs truncation | Table I counts describe *observed* clusters, while power truncation removes drawn clusters again | implemented as described; effect quantified (RESULTS.md, LIMITATIONS.md) |
| U1-U8 | unspecified | oversampling factor; KPD feature scaling/kernel/level-1 threshold; correction-table design; NCP threshold in Eq. 11; peak/cut-off coupling; omni/max-dir definitions (taken from [21]); Fig. 17 sample sizes and distances | METHODOLOGY.md |
