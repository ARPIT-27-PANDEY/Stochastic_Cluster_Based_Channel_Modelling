# Experiment 1 - generator performance

**Purpose.** Cost of the full 7-step generator on the native 1001x36x36 grid.

**Configuration.** see `config.json` (profile `standard`, seed `20260723`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- mean time per LoS realization 0.72 s, NLoS 2.19 s (single process, incl. metrics)
- time grows linearly with the batch size (no state is shared between realizations)
- NLoS is slower: ~90 clusters, each inverted and corrected inside the cut-off fixed point

## Comparison with the paper
- the paper reports no run times for its MATLAB toolkit; no comparison possible

## Limitations
- timings are machine-dependent (see env in config.json)
