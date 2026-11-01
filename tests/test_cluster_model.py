"""Cluster draws (steps 2-3) and the cluster power model (Eqs. 6-11)."""
import numpy as np
import pytest

from channel_model import power_model as PM
from channel_model.cluster_model import draw_cluster_count, draw_clusters
from channel_model.parameters import LOS_CLUSTER, PowerParams, load_channel_parameters

P = load_channel_parameters()


def test_cluster_count_conventions():
    rng = np.random.default_rng(0)
    los = np.array([draw_cluster_count("los", P, rng) for _ in range(20000)])
    nlos = np.array([draw_cluster_count("nlos", P, rng) for _ in range(20000)])
    assert np.all(los[:, 0] == 1)                    # one dominant LoS cluster
    assert los[:, 1].mean() == pytest.approx(3.39 * 3.35, abs=0.3)
    assert np.all(nlos[:, 0] == 0) and nlos[:, 1].min() >= 1
    assert (nlos[:, 1] - 1).mean() == pytest.approx(1.89 * 46.72, rel=0.03)
    with pytest.raises(ValueError):
        draw_cluster_count("foo", P, rng)


def test_draw_clusters_los():
    rng = np.random.default_rng(1)
    cl = draw_clusters("los", 20.0, P, rng)
    assert cl.cls[0] == LOS_CLUSTER
    assert cl.arrival_m[0] == 0 and cl.doa[0] == 0 and cl.dod[0] == 0
    assert np.all(cl.arrival_m >= 0)
    assert np.all((cl.s_doa > 0) & (cl.s_doa < 1)) and np.all((cl.s_dod > 0) & (cl.s_dod < 1))
    assert np.allclose(cl.ratio_db, 10 * np.log10((20 + cl.arrival_m) / 20))
    assert np.allclose(cl.power_db, cl.ncp_db - PM.fspl_db(20.0, P.fc_hz))


def test_draw_clusters_invalid_distance():
    with pytest.raises(ValueError):
        draw_clusters("los", 0.0, P, np.random.default_rng(0))


def test_fspl_value():
    # 20 log10(4 pi f / c) at 1 m and 145.5 GHz
    assert PM.fspl_db(1.0, 145.5e9) == pytest.approx(75.70, abs=0.01)


def test_equalization_weights_eq10():
    x = np.concatenate([np.full(90, 0.5), np.full(10, 15.5)])
    w = PM.equalization_weights(x, n_bins=20, x_range=(0, 20), min_fraction=0.01)
    assert w[0] == pytest.approx(100 / 20 / 90)
    assert w[-1] == pytest.approx(100 / 20 / 10)
    # a bin holding < 1 % of the samples keeps weight one
    x2 = np.concatenate([np.full(999, 0.5), [19.0]])
    assert PM.equalization_weights(x2)[-1] == 1.0


def test_llf_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 15, 300)
    y = -20 - 1.2 * x + 9 * rng.standard_normal(300)
    thr = np.full(300, -45.0)
    keep = y >= thr
    x, y, thr = x[keep], y[keep], thr[keep]
    w = rng.uniform(0.5, 2, x.size)
    th = np.array([-19.0, -1.0, 8.5, -0.1])
    g = PM.truncated_llf_grad(th, x, y, thr, w)
    h = 1e-6
    fd = [(PM.truncated_llf(th + h * e, x, y, thr, w) - PM.truncated_llf(th - h * e, x, y, thr, w)) / (2 * h)
          for e in np.eye(4)]
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-5)


def test_truncated_mle_recovers_parameters():
    """Ground truth -> truncation -> MLE; OLS is biased, the truncated MLE is not."""
    rng = np.random.default_rng(3)
    true = PowerParams(-20.0, -1.2, 9.0, -0.15)
    x = rng.uniform(0, 20, 6000)
    y = PM.sample_ncp(x, true, rng)
    thr = np.full_like(x, -40.0)
    keep = y >= thr
    x, y, thr = x[keep], y[keep], thr[keep]
    res = PM.fit_truncated_mle(x, y, thr, lr=1e-3, max_iter=20000, tol=1e-12, n_starts=2, rng=rng)
    lb, llf_lb = PM.fit_truncated_mle_lbfgs(x, y, thr)
    est = np.array([res.alpha, res.beta, res.rho, res.eta])
    assert np.allclose(est, [true.alpha, true.beta, true.rho, true.eta], atol=[1.2, 0.15, 0.8, 0.1])
    assert res.llf >= llf_lb - 0.5        # at least as good as the L-BFGS cross-check
    a_ols, b_ols = PM.fit_ols(x, y)
    # truncation removes weak clusters at large x -> OLS slope is biased upward (less negative)
    assert b_ols > res.beta + 0.1


def test_sigma_coverage_untruncated():
    rng = np.random.default_rng(4)
    p = PowerParams(0.0, 0.0, 5.0, 0.0)
    x = np.zeros(50000)
    y = PM.sample_ncp(x, p, rng)
    emp, exp_ = PM.sigma_coverage(x, y, np.full_like(x, -1e9), p)
    assert emp == pytest.approx(0.6827, abs=0.01) and exp_ == pytest.approx(0.6827, abs=1e-3)
