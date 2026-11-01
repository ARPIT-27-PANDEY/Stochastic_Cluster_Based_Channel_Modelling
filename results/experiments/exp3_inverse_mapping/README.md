# Experiment 3 - inverse mapping and discretisation correction

**Purpose.** Accuracy and robustness of kappa = F^-1(S_obs; DR) and of the pre-computed correction table.

**Configuration.** see `config.json` (profile `standard`, seed `20260726`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- continuous round trip: 100.0% converged, max relative kappa error 8.7e-10, 3.77 ms per cluster (vectorised)
- native grid, median |error| of re-observed spreads without correction: delay 0.128, DoA 0.100, DoD 0.104
- with correction: delay 0.016, DoA 0.018, DoD 0.019

## Comparison with the paper
- paper: Newton-Raphson 'converges reliably from a broad range of initialisations' - confirmed for the continuous map; paper gives no numbers for the correction, so only the qualitative claim (correction reduces discretisation error) can be checked

## Limitations
- the correction table is our own construction (grid, interpolation, table ranges) because the paper does not specify it; targets outside the table range are clipped
