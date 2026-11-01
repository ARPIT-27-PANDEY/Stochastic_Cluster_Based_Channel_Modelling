# Experiment 5 - KPD vs ML-KPD clustering

**Purpose.** Score both clustering algorithms against exact cluster labels of generated ADPSs (taps above 35 dB).

**Configuration.** see `config.json` (profile `standard`, seed `20260723`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- LOS: mean true clusters 9.5, KPD 3.3, ML-KPD 10.0; mean ARI KPD 0.776, ML-KPD 0.846
- NLOS: mean true clusters 29.7, KPD 7.7, ML-KPD 27.8; mean ARI KPD 0.448, ML-KPD 0.873

## Comparison with the paper
- the paper validates ML-KPD only visually on measured ADPSs (no accuracy numbers), so this is an additional, synthetic benchmark

## Limitations
- generated clusters are idealised (no noise, no sidelobes); on synthetic sounder data with IFFT sidelobes both KPD variants over-segment strong clusters (tests/test_adps.py, docs/LIMITATIONS.md)
- tiny clusters of 1-3 taps hidden under stronger ones cannot be separated by any density method
