# Implementation progress log

| Phase | Status | Notes |
|---|---|---|
| 1. Paper analysis | done | PAPER_ANALYSIS.md; 13 discrepancies + 8 unspecified details recorded |
| 2. Toolkit / data investigation | done | DATA_AVAILABILITY.md; toolkit form-gated (not submitted), data unpublished |
| 3. Environment | done | Python 3.12 venv (3.15 could not build PyYAML); pip freeze in results/logs |
| 4. Configuration | done | Table I primary, text values and modes configurable; published benchmarks in YAML |
| 5. ADPS (Eq. 1) + oversampling | done | Parseval and off-grid peak tests |
| 6. MPC extraction (35 dB) | done | |
| 7. KPD / ML-KPD | done | ARI ~1 on generated ADPSs; over-segmentation on un-windowed sounder test data documented |
| 8. Statistical model | done | |
| 9. Power model + truncated MLE | done | gradient FD-checked; recovery test; OLS bias reproduced |
| 10. Truncation map + regimes | done | brute-force verified (< 1 %) |
| 11. Inverse mapping | done | 3-D Newton diverged from poor starts -> reduced 2-D Newton + closed-form kappa_tau, 100 % convergence |
| 12. Discretisation correction | done | table design ours; bounded after a runaway was found (see below) |
| 13. Generator | done | explicit cut-off/peak fixed point; 2-3 sweeps |
| 14. Digitisation of published curves | done | Figs. 2, 6, 10, 11, 17 |
| 15. Validation (CvM, Table II) | done | scripts/validate_model.py |
| 16. Experiments 1-7 | done | results/experiments/ |
| 17. Figures 2-17 | done | results/figures/FIGURES.md |
| 18. Tests | done | 55 tests incl. integration, all passing |
| 19. Docs, notebooks, slides | done | |

## Problems met and how they were resolved
1. **Brute-force reference out of memory** (5.8 GiB tensor) - loop over delay slices.
2. **3-D Newton inversion diverged** from poor warm starts into saturated angular regions -
   exploit the separability (angular spreads independent of kappa_tau): 2-D Newton with
   complex-step Jacobian, bisection initialisation, closed-form kappa_tau.
3. **Cut-off 2-cycle** when two clusters have nearly equal peaks - damped cut-off update.
4. **Runaway correction in the generator** (kappa_tau up to 1.8 km, 940 s per NLoS channel): a
   damped correction carried over between sweeps pushed angular targets to the clip limit;
   at very narrow angular decay the grid keeps only the centre bin and the discrete delay
   spread exceeds the continuous one by tens of bins. Fix: per sweep, a fresh correction loop
   at the current DR_eff (as in `solve_decay_constants`) with |Delta| <= 50 % of the target.
   After the fix: 2-3 sweeps, 1-7 s per NLoS channel, median re-observation error ~2 %.
5. **Overlapping cluster tails** produced ~10^6 taps in NLoS channels - each cluster
   contributes only its own above-cut-off taps.
6. **ML-KPD with linear power weights** gave ARI 0.2 on LoS channels - exp(normalised power)
   weighting (He et al.) gives ~1.0.
7. **Synthetic sounder test data**: IFFT wrap-around of negative-delay sidelobes created a
   spurious cluster at 300 m - first path offset by 3 m in the test generator.
