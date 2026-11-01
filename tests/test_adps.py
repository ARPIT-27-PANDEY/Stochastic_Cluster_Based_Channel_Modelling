"""ADPS computation (Eq. 1), cluster ADPS (Eqs. 12-13), MPC extraction and clustering."""
import numpy as np
import pytest

from channel_model.adps import (DelayAngleGrid, adps_from_frequency_response, cluster_adps_kron, cluster_profiles,
                                maxdir_pdp, omni_pdp)
from channel_model.kpd_clustering import count_modes, kpd, ml_kpd
from channel_model.mpc_extraction import extract_taps, make_test_paths, synthetic_frequency_response

G = DelayAngleGrid()


def test_grid_constants():
    assert G.n_angle == 36 and G.shape == (1001, 36, 36)
    assert G.max_delay_m == pytest.approx(1001 * 0.29979)
    assert G.phi[G.n_angle // 2] == 0.0


def test_parseval_and_peak_location():
    nf = 64
    f = np.arange(nf)
    tau_bin = 10
    H = np.exp(-2j * np.pi * f * tau_bin / nf)[:, None, None] * np.ones((1, 2, 2))
    P = adps_from_frequency_response(H, oversampling=1)
    assert np.argmax(P[:, 0, 0]) == tau_bin
    assert P[:, 0, 0].sum() == pytest.approx(np.sum(np.abs(H[:, 0, 0]) ** 2))


def test_oversampling_recovers_off_grid_peak():
    nf = 64
    f = np.arange(nf)
    H = np.exp(-2j * np.pi * f * 10.5 / nf)[:, None, None]      # half-bin offset (worst case)
    p1 = adps_from_frequency_response(H, 1).max()
    p5 = adps_from_frequency_response(H, 5).max()
    p_true = nf  # |sum|^2 / nf for an on-grid path of unit amplitude
    assert p1 < 0.5 * p_true           # ~ -3.9 dB scalloping loss
    assert p5 > 0.9 * p_true


def test_kron_equals_outer_product():
    pdp, ar, at = cluster_profiles(2.0, 0.3, 0.5, G, 20, 5, 30)
    K = cluster_adps_kron(pdp, ar, at)
    O = pdp[:, None, None] * ar[None, :, None] * at[None, None, :]
    assert np.allclose(K, O)
    assert np.argmax(ar) == 5 and np.argmax(at) == 30 and pdp[0] == 1.0


def test_omni_and_maxdir_pdp():
    A = np.zeros((5, 3, 3))
    A[0, 0, 0] = 4.0
    A[1, 1, 1] = 3.0
    A[2, 1, 1] = 3.0
    assert omni_pdp(A).tolist() == [4, 3, 3, 0, 0]
    assert maxdir_pdp(A).tolist() == [0, 3, 3, 0, 0]


def test_extract_taps_threshold():
    A = np.zeros(G.shape)
    A[0, 18, 18] = 1.0
    A[3, 18, 18] = 10 ** (-3.4)
    A[4, 18, 18] = 10 ** (-3.6)
    t = extract_taps(A, G, 35.0)
    assert t.n == 2 and set(t.delay_idx.tolist()) == {0, 3}


def test_count_modes():
    x = np.zeros(60)
    x[10:14] = 1
    x[40:44] = 1
    assert count_modes(x, 5) == 2
    y = np.zeros(60)
    y[10:14] = 1
    y[15:19] = 1        # peak spacing <= window: merged
    assert count_modes(y, 5) == 1
    c = np.zeros(36)
    c[0] = c[35] = 1    # wraps around: one mode on the circle
    assert count_modes(c, 3, circular=True) == 1


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def test_sounder_pipeline_runs():
    """Eq.(1) -> 35 dB taps -> KPD / ML-KPD on synthetic sounder TEST DATA (pipeline check).

    Accuracy is not asserted here: un-windowed IFFT sidelobes make strong clusters look
    multimodal in delay, so (ML-)KPD over-segments them (reported in docs/LIMITATIONS.md).
    """
    rng = np.random.default_rng(5)
    paths = make_test_paths(rng, n_clusters=4, paths_per_cluster=10)
    H = synthetic_frequency_response(paths, G, rng=rng)
    for win in (None, "hann"):
        P = adps_from_frequency_response(H, oversampling=5, window=win)
        taps = extract_taps(P, G, 35.0)
        assert taps.n > 20
        res = ml_kpd(taps, n_angle=G.n_angle)
        assert res.labels.shape == (taps.n,) and res.n_clusters >= 4
        assert kpd(taps).shape == (taps.n,)


def test_ml_kpd_recovers_generated_clusters():
    """On generator output the true cluster of every tap is known exactly."""
    from channel_model import ChannelGenerator, load_channel_parameters
    gen = ChannelGenerator(load_channel_parameters())
    r = gen.generate(20.0, "los", np.random.default_rng(100), keep_adps=True)
    taps = extract_taps(r.adps, G, 35.0)
    truth = r.tap_labels[taps.delay_idx, taps.doa_idx, taps.dod_idx]
    res = ml_kpd(taps, n_angle=G.n_angle)
    assert _ari(truth, res.labels) > 0.9
