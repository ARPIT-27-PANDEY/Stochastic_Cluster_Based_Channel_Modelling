# Reproducibility

## Environment used for the reported results
* Windows 10, 4 CPU cores, Python 3.12.10 in `.venv`
* exact package versions: `results/logs/pip_freeze.txt` (numpy 2.5, scipy 1.18, pandas 3.1,
  matplotlib 3.11, scikit-learn 1.9, statsmodels 0.15, PyYAML 6.0, pytest 9.2, python-pptx 1.0)
* Python >= 3.11 required (Python 3.15 could not build PyYAML at the time of writing).

## Randomness
* One master seed per profile (`config/simulation_config.yaml`).
* Realization i of condition c in data set t uses `SeedSequence(seed, spawn_key=(c, t, i))`;
  results are identical for any number of worker processes and any execution order.
* Common random numbers are used across arms of the ablation/sensitivity experiments.

## Commands (order matters only for caching)
```bash
python scripts/generate_channels.py  --profile standard   # builds data/synthetic/standard_*.csv.gz (+ correction table)
python scripts/validate_model.py     --profile standard   # results/tables/cvm_validation.csv, pathloss_table2.csv
python scripts/run_experiments.py    --profile standard   # results/experiments/exp1..exp7
python scripts/reproduce_figures.py  --profile standard   # results/figures/*.png + FIGURES.md
python -m pytest                                           # test suite
python presentation/presentation_source/build_presentation.py
```
Figs. 12-13 overlay the MLE of experiment 4, so run the experiments before the figures.
The discretisation-correction table (`data/processed/discretization_correction.npz`) is
deterministic; delete it to rebuild (~2 min).

## Expected run times (4 cores)
| step | smoke | standard |
|---|---|---|
| generate_channels | ~3 min | ~1.5 h |
| validate_model | < 1 min | ~1 min |
| run_experiments | ~20 min | ~1.5 h |
| reproduce_figures | ~3 min | ~3 min |
| pytest | ~3 min | - |

## Determinism checks
`tests/test_integration.py::test_nlos_realization_and_determinism` regenerates a channel with the
same seed and requires identical metrics.
