"""Builds notebooks/01-04 (run: python notebooks/_build_notebooks.py; then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SETUP = """import sys, pathlib
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from channel_model import *
P = load_channel_parameters()"""

NB = {
"01_statistical_model.ipynb": [
 ("md", "# 01 - Table I statistical model\nAnalytic distributions with the published parameters and samples from our implementation (simulated, no measured data)."),
 ("code", SETUP),
 ("code", """from channel_model import distributions as D
rng = np.random.default_rng(0)
n_los = 1 + D.round_cluster_count(D.gamma_sample(P.n_los_additional.shape, P.n_los_additional.scale, 20000, rng))
n_nlos = 1 + D.round_cluster_count(D.gamma_sample(P.n_nlos_minus_one.shape, P.n_nlos_minus_one.scale, 20000, rng))
print('mean cluster number LoS %.2f (paper 11.36), NLoS %.2f (paper 88.30)' % (n_los.mean(), n_nlos.mean()))"""),
 ("code", """m = P.doa_centroid['los']; x = np.linspace(-np.pi, np.pi, 721)
for mode in ('truncated', 'wrapped'):
    plt.semilogy(x, D.bimodal_laplace_pdf(x, m.weights, m.centers, m.decays, mode), label=mode)
plt.xlabel('DoA centroid [rad]'); plt.ylabel('pdf'); plt.legend(); plt.title('Eq. (3), LoS DoA'); plt.show()"""),
 ("code", """from channel_model.power_model import ncp_mean, ncp_std
x = np.linspace(0, 20, 50)
for k in ('los', 'nlos'):
    plt.plot(x, ncp_mean(x, P.power[k]), label=k); plt.fill_between(x, ncp_mean(x, P.power[k]) - ncp_std(x, P.power[k]), ncp_mean(x, P.power[k]) + ncp_std(x, P.power[k]), alpha=.2)
plt.xlabel('10log10(d/d_LoS) [dB]'); plt.ylabel('NCP [dB]'); plt.legend(); plt.show()"""),
],
"02_dynamic_range_inverse_mapping.ipynb": [
 ("md", "# 02 - Dynamic-range truncation and inverse mapping (Sec. VI)\nDeterministic computations: continuous map F(kappa; DR), its inverse, and the discrete-grid error."),
 ("code", SETUP),
 ("code", """from channel_model.dynamic_range import continuous_observables, discrete_observables, AngleGrid, regime_boundaries_db
dr = np.arange(1, 51.)
o = continuous_observables(10.0, 0.3*np.pi, 0.2*np.pi, dr)
print('regime boundaries [dB]:', np.round(regime_boundaries_db(0.3*np.pi, 0.2*np.pi), 2))
inv = invert_spreads(np.column_stack([o['s_tau'], o['s_r'], o['s_t']]), dr)
print('max relative kappa error:', np.abs(inv.kappa / [10, 0.3*np.pi, 0.2*np.pi] - 1).max())"""),
 ("code", """g = discrete_observables(10/0.29979, 0.3*np.pi, 0.2*np.pi, dr, AngleGrid(10.), 0.29979)
plt.plot(dr, o['s_tau'], label='continuous'); plt.plot(dr, g['s_tau'], ':', label='1 ns / 10 deg grid')
plt.xlabel('DR [dB]'); plt.ylabel('delay spread [m]'); plt.legend(); plt.show()"""),
],
"03_channel_generation_validation.ipynb": [
 ("md", "# 03 - Synthetic channels and validation\nGenerates a few channels and reads the validation tables written by `scripts/validate_model.py` (run it first)."),
 ("code", SETUP),
 ("code", """gen = ChannelGenerator(P, correction=DiscretizationCorrection.load(ROOT / 'data/processed/discretization_correction.npz'))
r = gen.generate(20.0, 'los', np.random.default_rng(1), keep_adps=True)
print(r.n_visible, 'of', r.clusters.n, 'clusters observable;', r.metrics)
plt.imshow(10*np.log10(np.maximum(r.adps.max(axis=2)[:60], 1e-30) / r.adps.max()), vmin=-35, vmax=0, aspect='auto')
plt.xlabel('DoA bin'); plt.ylabel('delay bin'); plt.colorbar(label='dB'); plt.show()"""),
 ("code", """cvm = pd.read_csv(ROOT / 'results/tables/cvm_validation.csv')
cvm[['metric', 'condition', 'published_cvm', 'recomputed_paper_cvm', 'ours_vs_measured_cvm', 'null_cvm_95', 'status']]"""),
 ("code", """pd.read_csv(ROOT / 'results/tables/pathloss_table2.csv')[['case', 'parameter', 'published_synthetic', 'ours_median_over_repeats', 'ours_p2.5', 'ours_p97.5']]"""),
],
"04_clustering.ipynb": [
 ("md", "# 04 - ML-KPD clustering\nClusters a generated ADPS (exact ground truth available) and a synthetic-sounder test ADPS (TEST DATA)."),
 ("code", SETUP),
 ("code", """from channel_model.mpc_extraction import extract_taps
from channel_model.kpd_clustering import kpd, ml_kpd
from sklearn.metrics import adjusted_rand_score as ari
gen = ChannelGenerator(P)
r = gen.generate(20.0, 'los', np.random.default_rng(100), keep_adps=True)
taps = extract_taps(r.adps, gen.grid, 35.0)
truth = r.tap_labels[taps.delay_idx, taps.doa_idx, taps.dod_idx]
res = ml_kpd(taps)
print('taps', taps.n, 'true', len(np.unique(truth)), 'ML-KPD', res.n_clusters, 'ARI', round(ari(truth, res.labels), 3), 'KPD ARI', round(ari(truth, kpd(taps)), 3))"""),
 ("code", """plt.scatter(np.rad2deg(taps.doa), taps.tau_m, c=res.labels, s=8, cmap='tab20')
plt.xlabel('DoA [deg]'); plt.ylabel('delay [m]'); plt.title('ML-KPD labels (generated channel)'); plt.show()"""),
],
}
for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
print("written", list(NB))
