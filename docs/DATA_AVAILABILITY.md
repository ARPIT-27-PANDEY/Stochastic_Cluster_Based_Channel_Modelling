# Data and toolkit availability

## 1. Measurement data
The paper's model is extracted from 145-146 GHz double-directional VNA measurements
(21 LoS and 17 NLoS Tx-Rx pairs, USC campus; campus geometry in Fig. 1 of the paper,
campaign described in [21], arXiv:2109.13693).

* The raw frequency responses, the ADPSs, the extracted MPC taps and the cluster lists
  are **not published** with the paper or with [21]. We found no public repository
  (IEEE DataPort, GitHub, Zenodo, the WiDeS web site) holding them.
* Consequence: every step that needs measurements - clustering the measured ADPSs,
  fitting Table I, the measured curves of Figs. 2-13, the measured side of Fig. 17 and
  of Table II - **cannot be re-run**. We do not invent measurement data.

## 2. Authors' toolkit ([60])
* Location: https://wides.usc.edu/research_matlab.html, entry "Channel Generation
  Toolkit for sub-THz Outdoor D2D Links at 145 GHz".
* Access: the download is gated by Google Forms (forms.gle/AVihbxiwyFmjHkZ39 and
  forms.gle/Wo9sjpoASWnBt8kD6) asking for personal and institutional details.
* We did **not** submit the form: doing so means entering personal data on the
  contributors' behalf, and the licence/terms are not stated on the page. A team
  member can request it manually; the toolkit would then allow a direct numerical
  comparison of generators (planned as future work, see LIMITATIONS.md).
* Nothing in this repository is derived from the toolkit's code. The implementation is
  written from the paper's equations and text only.

## 3. What we used instead
| Source | What | How it is used | Label in outputs |
|---|---|---|---|
| Paper, Table I | all statistical parameters | `config/channel_parameters.yaml` (+ in-text alternatives) | "Table I model" |
| Paper, Table II and text | path-loss and coverage numbers, CvM values | `published_benchmarks` in the same YAML | "published" |
| [21], Table II | the 21 LoS / 17 NLoS link distances | distances of the validation channels | - |
| Paper, Figs. 2, 6, 10, 11, 17 | CDF curves | digitised from the PDF's vector graphics | "measured (digitized)", "paper synthetic (digitized)" |

### Digitisation (scripts/digitize_published_figures.py)
The figures are vector graphics. PyMuPDF reads the drawing commands of each panel; the
axis frame is located from its dark lines and the known axis ranges map page to data
coordinates (log axes where the paper uses them). Step CDFs of the measured data
(21 / 17 samples) are converted back to samples (one sample per vertical step; the
sample count is inferred from the step height). The "paper synthetic" curves of Fig. 17
have 205 (LoS) and 40-50 (NLoS) steps. Three LoS synthetic samples in Fig. 17b lie
beyond the plotted axis and were placed at the axis limit.

Checks: the digitised *fitted* curves of Figs. 2/6/10/11 agree with the Table I CDFs
within 0.0013-0.015 (max |dCDF|), except the very steep LoS-cluster DoA spread curve
(0.047, horizontal position error of a near-vertical line). The digitised measured cluster
counts have mean 11.4 (LoS) and 88.1 (NLoS), matching the paper's means 11.36 and 88.30.
Re-computing the paper's own CvM values from the digitised Fig. 17 curves gives
0.0057/0.0046 (delay LoS/NLoS), 0.0089/0.0153 (DoA), 0.0043/0.0228 (DoD) against the
published 0.004/0.007, 0.007/0.020, 0.003/0.029 - same magnitude, which also confirms our
reading of the CvM normalisation (METHODOLOGY.md section 10).

These curves are **reconstructions of plotted results**, not measurement data, and are
labelled as such everywhere. Only derived CSVs are stored (`data/processed/digitized/`);
the PDF is not part of the repository.

## 4. Repository data folders
* `data/raw/` - empty on purpose (no raw measurements available).
* `data/processed/` - digitised curves and the discretisation-correction table
  (computed by us, `discretization_correction.npz`).
* `data/synthetic/` - generated data sets (CSV, regenerable with `scripts/generate_channels.py`).
