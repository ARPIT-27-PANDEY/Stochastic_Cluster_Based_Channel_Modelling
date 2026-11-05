# sub-thz-d2d-channel-model

GNU Radio reproduction (with a supporting Python channel-model library) of

> Y. Cheng, N. A. Abbasi, J. Gomez-Ponce, Z. Zhang, A. F. Molisch, **"Stochastic Cluster-Based
> Channel Modeling for Sub-Terahertz Outdoor Device-to-Device Radio Propagation at 145 GHz"**,
> IEEE Transactions on Wireless Communications, vol. 25, pp. 20883-20899, 2026,
> doi:10.1109/TWC.2026.3714332.

## Contributors - Group 10
| Enrolment no. | Name |
|---|---|
| 23116075 | Piyush Bhaskar |
| 23116005 | Akersheet Tripathi |
| 23116017 | Arpit Kumar Pandey |

## What is in here
* **GNU Radio implementation** (`gnuradio/`, GNU Radio 3.10): a no-GUI VNA channel-sounder
  flowgraph (cluster-model channel -> 1001-tone sweep per horn orientation -> noise -> IFFT ->
  ADPS -> 35 dB taps -> spreads and path gains) and a QT GUI channel emulator; see gnuradio/README.md.
  `scripts/run_gnuradio.py` runs the sounder in parallel and compares its output with the paper
  (results/gnuradio/);
* a **calibrated parameter set** (`scripts/calibrate_model.py` ->
  `config/channel_parameters_calibrated.yaml`) that brings the generated NLoS spreads close to the
  paper's synthetic Fig. 17 curves. It is a fit to the paper's published outputs, labelled
  "calibrated" everywhere; the published Table I is never changed and its results are always shown too;
* the complete measurement-side processing chain (ADPS of Eq. 1 with delay oversampling, 35 dB
  MPC taps, KPD and multi-level KPD clustering);
* the Table I statistical cluster model, the truncation-aware cluster-power MLE (Eqs. 10-11);
* the dynamic-range truncation map F(kappa; DR), its Newton inverse, a discretisation-correction
  table and the full 7-step synthetic channel generator on the native 1 ns x 10 deg x 10 deg grid;
* validation against digitised curves of the paper (Fig. 17 / CvM) and the path-loss models of Table II;
* seven experiments, figures 2-17 at 300 dpi, a pytest suite, notebooks and a slide deck.

**Data.** The measurement data and the authors' MATLAB toolkit are not publicly available
(docs/DATA_AVAILABILITY.md). Nothing here is measured data: simulated results use the published
parameters, and published curves were digitised from the paper's vector figures and are
labelled "digitized" everywhere. The paper PDF is not included (copyright).

## Quick start
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest                               # ~3 min
.venv/Scripts/python scripts/generate_channels.py --profile smoke
.venv/Scripts/python scripts/validate_model.py --profile smoke
.venv/Scripts/python scripts/run_experiments.py --profile smoke
.venv/Scripts/python scripts/reproduce_figures.py --profile smoke
```
Profiles: `smoke` (minutes), `standard` (600 channels per condition, about 1-2 h on 4 cores),
`full` (3000 channels per condition). Seeds are fixed in `config/simulation_config.yaml`.
The digitised curves are already in `data/processed/digitized/`; to redo them you need the PDF:
`python scripts/digitize_published_figures.py --pdf <paper.pdf>`.

### Use the generator
```python
import numpy as np
from channel_model import ChannelGenerator, DiscretizationCorrection, load_channel_parameters
gen = ChannelGenerator(load_channel_parameters(),
                       correction=DiscretizationCorrection.load("data/processed/discretization_correction.npz"))
r = gen.generate(d_los_m=20.0, condition="los", rng=np.random.default_rng(1), keep_adps=True)
r.adps.shape          # (1001, 36, 36) delay x DoA x DoD, linear power, 35 dB cut-off applied
r.metrics             # delay / angular spreads, omni and max-dir path gain, number of taps
```

## Layout
```
gnuradio/      GRC flowgraphs (sounder, emulator), their generator and screenshot scripts
config/        channel_parameters.yaml (Table I + paper alternatives + published benchmarks), simulation_config.yaml
src/channel_model/  parameters, distributions, adps, mpc_extraction, kpd_clustering, cluster_model, power_model,
               dynamic_range, inverse_mapping, channel_generator, channel_metrics, path_loss, validation, digitize
scripts/       generate_channels, validate_model, run_experiments, reproduce_figures, digitize_published_figures,
               run_gnuradio, calibrate_model
tests/         pytest suite (distributions, cluster model, ADPS/clustering, dynamic range, inverse mapping,
               validation, integration)
notebooks/     01-04 walkthroughs
results/       figures (FIGURES.md catalogue), tables, experiments/<exp>/README.md, logs
docs/          PAPER_ANALYSIS, DATA_AVAILABILITY, METHODOLOGY, IMPLEMENTATION_PROGRESS, RESULTS, REPRODUCIBILITY, LIMITATIONS
presentation/  presentation.pptx and its python-pptx source
```

## Main findings (details: docs/RESULTS.md)
* The truncation map, its inverse and Figs. 14-15 are reproduced to numerical precision; the
  paper's four regime boundaries (14.48 / 21.71 / 36.19 dB) follow analytically.
* LoS channels generated from Table I match the measured delay/DoA/DoD spread CDFs within
  sampling noise; NLoS channels are more dispersive than measured.
* The truncation-aware MLE recovers the true NCP slope on simulated data where OLS is biased, as
  the paper argues.
* Several inconsistencies in the paper (Eq. 4, Eq. 10, Table I vs text, Fig. 14/15 labels) are
  documented with the choice made for each.

## Licence
Code: MIT. The reproduced paper, its figures and data belong to their authors/IEEE.
