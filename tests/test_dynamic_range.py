"""Dynamic-range truncation of a cluster (Sec. VI, Eq. 14, Figs. 14-15)."""
import numpy as np
import pytest

from channel_model import dynamic_range as DR
from channel_model.distributions import fleury_angular_spread


def _brute(kt, ka, kb, dr_db, n_tau=1500, n_ang=721):
    """Riemann-sum reference, looping over delay slices to bound memory."""
    L = dr_db * np.log(10) / 10
    tau = np.linspace(0, kt * L, n_tau)
    phi = np.linspace(-np.pi, np.pi, n_ang)
    ang = np.abs(phi)[:, None] / ka + np.abs(phi)[None, :] / kb
    m0 = m1 = m2 = 0.0
    ar = np.zeros(n_ang)
    at = np.zeros(n_ang)
    for t in tau:
        e = t / kt + ang
        w = np.exp(-e) * (e <= L)
        s = w.sum()
        m0, m1, m2 = m0 + s, m1 + s * t, m2 + s * t * t
        ar += w.sum(axis=1)
        at += w.sum(axis=0)
    mt = m1 / m0
    return np.sqrt(m2 / m0 - mt**2), fleury_angular_spread(phi, ar), fleury_angular_spread(phi, at)


@pytest.mark.parametrize("dr", [5.0, 20.0, 40.0])
def test_continuous_matches_brute_force(dr):
    kt, ka, kb = 3.0, 0.4, 0.7
    o = DR.continuous_observables(kt, ka, kb, dr)
    st, sr, sd = _brute(kt, ka, kb, dr)
    assert o["s_tau"][0] == pytest.approx(st, rel=0.01)
    assert o["s_r"][0] == pytest.approx(sr, rel=0.01)
    assert o["s_t"][0] == pytest.approx(sd, rel=0.01)


def test_infinite_dr_limits():
    o = DR.continuous_observables(10.0, 0.3, 0.6, 400.0)
    assert o["s_tau"][0] == pytest.approx(10.0, rel=1e-6)
    assert o["s_r"][0] == pytest.approx(DR.infinite_dr_angular_spread(0.3), rel=1e-6)
    assert o["s_t"][0] == pytest.approx(DR.infinite_dr_angular_spread(0.6), rel=1e-6)


def test_angular_spreads_independent_of_kappa_tau():
    a = DR.continuous_observables(1.0, 0.3, 0.5, 18.0)
    b = DR.continuous_observables(50.0, 0.3, 0.5, 18.0)
    assert a["s_r"][0] == pytest.approx(b["s_r"][0], rel=1e-10)
    assert b["s_tau"][0] / a["s_tau"][0] == pytest.approx(50.0, rel=1e-10)


def test_spreads_increase_with_dr():
    dr = np.arange(1, 51, dtype=float)
    o = DR.continuous_observables(10.0, 0.2 * np.pi, 0.3 * np.pi, dr)
    for k in ("s_tau", "s_r", "s_t"):
        assert np.all(np.diff(o[k]) > -1e-12)


def test_eq14_delay_only_limit():
    # with very wide (flat) angular profiles the joint cut-off reduces to Eq. (14) at tau_tr = kappa L
    x = DR.eq14_delay_spread(1.0, 1e6)
    assert x == pytest.approx(1.0)
    assert DR.eq14_delay_spread(1.0, 1.0) < 0.3


def test_regime_boundaries_fig14():
    b = DR.regime_boundaries_db(0.2 * np.pi, 0.3 * np.pi)
    assert np.allclose(b, (14.48, 21.71, 36.19), atol=0.01)


def test_discrete_converges_to_continuous_on_fine_grid():
    kt, ka, kb, dr = 5.0, 0.5, 0.8, 25.0
    res = 0.005
    d = DR.discrete_observables(kt / res, ka, kb, dr, DR.AngleGrid(0.25), res)
    c = DR.continuous_observables(kt, ka, kb, dr)
    for k in ("s_tau", "s_r", "s_t"):
        assert d[k][0] == pytest.approx(c[k][0], rel=0.02)


def test_discrete_against_explicit_grid():
    g = DR.AngleGrid(10.0)
    kb, ka, kt, dr = 4.0, 0.3, 0.5, 20.0
    d = DR.discrete_observables(kb, ka, kt, dr, g, 1.0)
    L = dr * np.log(10) / 10
    k = np.arange(400)
    e = k[:, None, None] / kb + np.abs(g.phi)[None, :, None] / ka + np.abs(g.phi)[None, None, :] / kt
    w = np.exp(-e) * (e <= L + 1e-12)
    assert d["shape_sum"][0] == pytest.approx(w.sum(), rel=1e-9)
    assert d["n_taps"][0] == int((w > 0).sum())
    m = (w.sum(axis=(1, 2)) * k).sum() / w.sum()
    assert d["mean_tau"][0] == pytest.approx(m, rel=1e-9)


def test_effective_dynamic_range():
    assert np.allclose(DR.effective_dynamic_range([0.0, -10.0, -40.0], -35.0), [35.0, 25.0, -5.0])
