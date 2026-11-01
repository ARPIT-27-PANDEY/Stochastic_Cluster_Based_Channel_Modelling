"""End-to-end generator (Fig. 16 steps 1-7) on the native measurement grid."""
from pathlib import Path

import numpy as np
import pytest

from channel_model import ChannelGenerator, DiscretizationCorrection, load_channel_parameters
from channel_model.channel_generator import ld_setting_visible
from channel_model.mpc_extraction import extract_taps

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "discretization_correction.npz"


@pytest.fixture(scope="module")
def gen():
    corr = DiscretizationCorrection.load(TABLE) if TABLE.exists() else None
    return ChannelGenerator(load_channel_parameters(), correction=corr)


def test_los_realization(gen):
    r = gen.generate(20.0, "los", np.random.default_rng(11), keep_adps=True)
    assert r.adps.shape == (1001, 36, 36)
    assert r.visible[0]                                   # LoS cluster always observable
    peak = r.adps.max()
    nz = r.adps[r.adps > 0]
    assert 10 * np.log10(peak / nz.min()) <= 35.0 + 1e-6  # global 35 dB cut-off applied
    assert 10 * np.log10(peak) == pytest.approx(r.peak_db.max(), abs=1e-6)
    assert np.all(np.isfinite(r.kappa[r.visible]))
    assert r.metrics["n_taps"] == extract_taps(r.adps, gen.grid, 35.0).n
    # each visible cluster re-observed on the grid reproduces its drawn spreads
    rel = r.reobserved["s_tau"][r.visible] / r.clusters.s_tau_m[r.visible]
    assert np.median(np.abs(rel - 1)) < 0.1


def test_nlos_realization_and_determinism(gen):
    a = gen.generate(40.0, "nlos", np.random.default_rng(12), compute_metrics=True)
    b = gen.generate(40.0, "nlos", np.random.default_rng(12), compute_metrics=True)
    assert a.n_visible >= 1
    assert a.metrics == b.metrics
    assert np.nanmax(a.kappa[:, 1:]) < 100       # no runaway decay constants
    assert a.fixed_point_iterations < gen.max_fp_iter


def test_ld_setting(gen):
    r = gen.generate(30.0, "los", np.random.default_rng(13), compute_metrics=False)
    m = ld_setting_visible(r, 15.0)
    assert not m[0]
    assert m.sum() <= (r.visible & (r.clusters.cls != 0)).sum()


def test_invalid_inputs(gen):
    with pytest.raises(ValueError):
        ChannelGenerator(load_channel_parameters(), dynamic_range_db=0)
    with pytest.raises(ValueError):
        gen.generate(10.0, "indoor", np.random.default_rng(0))
