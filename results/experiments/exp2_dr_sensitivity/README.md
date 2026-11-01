# Experiment 2 - dynamic-range sensitivity

**Purpose.** How the observable channel statistics depend on the DR used to generate (and observe) the channels.

**Configuration.** see `config.json` (profile `standard`, seed `20260723`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- LOS: observable clusters 1.0 -> 7.4, median DS 0.32 -> 2.57 m, median DoA spread 0.140 -> 0.204 for DR 5 -> 35 dB
- NLOS: observable clusters 2.9 -> 60.4, median DS 0.15 -> 22.82 m, median DoA spread 0.093 -> 0.928 for DR 5 -> 35 dB

## Comparison with the paper
- the paper uses 35 dB throughout and argues (Sec. V-A, LD setting) that DR changes cluster counts; the monotone growth of observable clusters and spreads with DR is consistent with that argument

## Limitations
- the Table I statistics were extracted at 35 dB; using them at other DRs is an extrapolation (the inverse mapping keeps per-cluster spreads at their drawn values for every DR)
