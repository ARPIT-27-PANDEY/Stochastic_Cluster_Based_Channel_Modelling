# Experiment 7 - sensitivity to ambiguous published parameters

**Purpose.** Table I vs in-text arrival-time Gamma parameters, and normalised vs wrapped bimodal Laplace (Eq. 3).

**Configuration.** see `config.json` (profile `standard`, seed `20260723`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- LOS table_I: DS 2.78 m, ASA 0.202, ASD 0.232, CvM vs measured delay/DoA/DoD 0.0135/0.0285/0.0068
- LOS text_arrival: DS 2.93 m, ASA 0.202, ASD 0.230, CvM vs measured delay/DoA/DoD 0.0119/0.0298/0.0057
- LOS wrapped_laplace: DS 2.87 m, ASA 0.216, ASD 0.228, CvM vs measured delay/DoA/DoD 0.0201/0.0195/0.0051
- NLOS table_I: DS 20.92 m, ASA 0.877, ASD 0.872, CvM vs measured delay/DoA/DoD 0.0191/0.0831/0.1300
- NLOS text_arrival: DS 20.94 m, ASA 0.890, ASD 0.875, CvM vs measured delay/DoA/DoD 0.0209/0.0809/0.1296

## Comparison with the paper
- the paper does not state which arrival parameters its toolkit uses; differences between variants indicate how much the ambiguity matters

## Limitations
- same seeds across variants
