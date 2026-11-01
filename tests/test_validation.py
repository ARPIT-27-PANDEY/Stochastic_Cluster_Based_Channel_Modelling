"""CvM validation statistics and path-loss fitting (Secs. VII-VIII)."""
import numpy as np
import pytest
from scipy import stats

from channel_model.path_loss import ci_overlap, fit_alpha_beta, fit_close_in
from channel_model.power_model import fspl_db
from channel_model.validation import bootstrap_cvm_null, cvm_distance, cvm_one_sample, cvm_T


def test_cvm_identical_and_disjoint():
    a = np.arange(10.0)
    assert cvm_distance(a, a) == 0.0
    d = cvm_distance(np.arange(10.0), np.arange(10.0) + 100)
    assert d > 0.3


def test_cvm_T_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=21), rng.normal(0.3, 1.2, size=205)
    assert cvm_T(a, b) == pytest.approx(stats.cramervonmises_2samp(a, b).statistic, rel=1e-9)


def test_cvm_one_sample():
    rng = np.random.default_rng(1)
    x = rng.random(400)
    assert cvm_one_sample(x, lambda u: u) < 0.005


def test_bootstrap_null_scale():
    null = bootstrap_cvm_null(21, 205, 300, np.random.default_rng(2))
    # E[T] = 1/6 asymptotically -> E[omega^2] ~ (1/6) (N+M)/(NM)
    assert np.mean(null) == pytest.approx((1 / 6) * (21 + 205) / (21 * 205), rel=0.2)


def test_alpha_beta_recovery_and_ci():
    rng = np.random.default_rng(3)
    d = rng.uniform(1, 100, 50)
    pl = 75.7 + 10 * 2.0 * np.log10(d) + 2.0 * rng.standard_normal(50)
    f = fit_alpha_beta(d, pl)
    assert f.alpha_ci[0] < f.alpha < f.alpha_ci[1]
    assert abs(f.beta - 2.0) < 0.3
    assert f.sigma_ci[0] < f.sigma < f.sigma_ci[1]


def test_sigma_ci_reproduces_table2_width():
    # Table II, LoS omni: sigma 1.98 -> [1.66, 2.47] with N = 50, dof = 48
    sse = 1.98**2 * 48
    lo = np.sqrt(sse / stats.chi2.ppf(0.975, 48))
    hi = np.sqrt(sse / stats.chi2.ppf(0.025, 48))
    assert lo == pytest.approx(1.66, abs=0.015) and hi == pytest.approx(2.47, abs=0.01)


def test_close_in_exact():
    d = np.array([2.0, 5, 10, 30, 80])
    pl = fspl_db(1.0, 145.5e9) + 10 * 2.3 * np.log10(d)
    f = fit_close_in(d, pl, 145.5e9)
    assert f.n_exp == pytest.approx(2.3, abs=1e-10)
    assert f.sigma == pytest.approx(0.0, abs=1e-9)


def test_ci_overlap():
    assert ci_overlap((0, 1), (0.5, 2)) and not ci_overlap((0, 1), (1.5, 2))
