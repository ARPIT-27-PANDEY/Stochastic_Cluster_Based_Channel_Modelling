# Experiment 6 - discretisation-correction ablation

**Purpose.** Effect of step 5's correction on how faithfully generated clusters reproduce their drawn spreads on the grid.

**Configuration.** see `config.json` (profile `standard`, seed `20260723`). Results: `results.json` and the CSV/PNG files here.

**Data.** Simulated with the published Table I parameters (no measured data).

## Observations (computed from this run)
- LOS correction=True: median |re-observed/target-1| delay 0.037, DoA 0.048, DoD 0.045; channel median DS 2.78 m
- LOS correction=False: median |re-observed/target-1| delay 0.215, DoA 0.099, DoD 0.098; channel median DS 2.51 m
- NLOS correction=True: median |re-observed/target-1| delay 0.017, DoA 0.025, DoD 0.025; channel median DS 20.92 m
- NLOS correction=False: median |re-observed/target-1| delay 0.182, DoA 0.084, DoD 0.086; channel median DS 20.48 m

## Comparison with the paper
- paper states that without correction discretisation error is 'particularly pronounced' for small spreads; no numbers given

## Limitations
- same random numbers in both arms (common seeds)
