# Limitations

## Data
1. **No measurement data.** Table I cannot be re-extracted; the measured histograms/scatter
   plots of Figs. 3-5, 7-9 and 12-13 cannot be compared. For those figures we plot the
   published model and our simulated draws only.
2. **Digitised curves** (Figs. 2, 6, 10, 11, 17) carry reading errors: <= 0.015 in CDF for
   most curves (0.047 for the near-vertical LoS-cluster DoA curve). The Fig. 17 sample sizes
   were inferred from step heights; three LoS synthetic DoA samples beyond the axis were set
   to the axis limit.
3. **Authors' toolkit not used** (form-gated). Implementation choices could therefore not be
   checked against the reference code.

## Unspecified details (our choices, METHODOLOGY.md)
4. Delay oversampling factor (5), KPD feature scaling, kernel, level-1 connectivity threshold (0.5).
5. Design of the discretisation-correction table and the 50 % bound on the correction.
6. Coupling between cluster peaks, shapes and the global cut-off (explicit fixed point here).
7. The NCP truncation level used in Eq. 11 (per-realization cut-off + FSPL in our synthetic tests).
8. The link distances and sample sizes of the paper's validation channels (we use the 21/17
   measured distances of [21]).

## Model-level observations
9. **NLoS dispersion.** NLoS channels generated from Table I are more dispersive (median delay
   spread, DoA/DoD spreads) than the measured CDFs and the authors' own synthetic CDFs in Fig. 17.
   With ~90 clusters per channel, uniform centroids and an NCP standard deviation of ~10 dB, many
   off-boresight clusters remain within 35 dB of the strongest one. We did not tune any
   parameter to remove this; possible causes are an unstated detail of the authors' generator
   (e.g. how the drawn count relates to observable clusters) or our reading of the power model.
10. **Cluster count vs truncation (D13).** Table I counts were fitted to observed clusters, but in
   the generator a part of the drawn clusters falls below the cut-off, so fewer clusters are
   observable than measured (Fig. 2).
11. **LoS path loss** comes out ~3 dB lower than [21] over 1-100 m (intercept 72.2 vs 76.9 dB, slope
   1.96 vs 1.77; the alpha-beta CIs still overlap in 80 % of repeats, the narrow CI-model exponent does not); the LoS cluster NCP (7.63 dB above FSPL) is spread over several orientation pairs,
   and the omni PDP keeps only the strongest one per delay bin.
12. Extrapolating Table I (extracted at 35 dB) to other DRs (experiment 2) is not validated.

## Clustering
13. On synthetic sounder data without a frequency window, IFFT sidelobes make strong clusters
   look multimodal in delay and KPD/ML-KPD over-segment them (ARI 0.3-0.6; a Hann window helps
   in some cases). The test suite checks only that this pipeline runs; accuracy is measured on
   generator output, where ground truth is exact.
14. Weak clusters lying under stronger ones (a few taps) cannot be separated by any density-based method.

## Computation
15. The standard profile uses 600 channels per condition (paper: unknown). The full profile
   (3000) was not run for the reported numbers because of run time.
16. Truncated-MLE bias on simulated LoS data (exp. 4): the generator's visibility rule is a peak-tap
   criterion, so Eq. 11's sharp NCP threshold does not hold exactly.
17. The truncated MLE uses a reduced iteration budget outside the full profile (convergence is
   reported per start in experiment 4).
