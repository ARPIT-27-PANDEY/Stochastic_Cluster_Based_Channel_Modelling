"""Statistical distributions of Table I and the spread definitions (Eqs. 2-4, 7)."""
import numpy as np
import pytest
from scipy import integrate, stats

from channel_model import distributions as D
from channel_model.parameters import load_channel_parameters


def test_wrap_angle_range():
    x = np.linspace(-20, 20, 1001)
    w = D.wrap_angle(x)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.exp(1j * w), np.exp(1j * x))


def test_gamma_moments():
    rng = np.random.default_rng(0)
    s = D.gamma_sample(3.39, 3.35, 200_000, rng)
    assert s.mean() == pytest.approx(3.39 * 3.35, rel=0.01)
    assert s.var() == pytest.approx(3.39 * 3.35**2, rel=0.03)
    x = np.linspace(0.1, 40, 50)
    assert np.allclose(D.gamma_cdf(x, 3.39, 3.35), stats.gamma.cdf(x, 3.39, scale=3.35))


def test_lognormal_is_natural_log():
    rng = np.random.default_rng(1)
    s = D.lognormal_sample(-1.42, 0.52, 200_000, rng)
    assert np.mean(np.log(s)) == pytest.approx(-1.42, abs=0.01)
    assert np.std(np.log(s)) == pytest.approx(0.52, abs=0.01)


def test_round_cluster_count():
    assert D.round_cluster_count([0.49, 0.5, 1.5, 2.4999]).tolist() == [0, 1, 2, 2]


@pytest.mark.parametrize("mode", ["truncated", "wrapped"])
def test_bimodal_laplace_pdf_normalised(mode):
    p = load_channel_parameters()
    m = p.doa_centroid["los"]
    val, _ = integrate.quad(lambda x: D.bimodal_laplace_pdf(x, m.weights, m.centers, m.decays, mode), -np.pi, np.pi,
                            points=[0.0], limit=400)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_bimodal_laplace_sampler_matches_pdf():
    p = load_channel_parameters()
    m = p.dod_centroid["los"]
    rng = np.random.default_rng(2)
    s = D.bimodal_laplace_sample(m.weights, m.centers, m.decays, 100_000, rng, mode="truncated")
    grid = np.linspace(-np.pi, np.pi, 4001)
    pdf = D.bimodal_laplace_pdf(grid, m.weights, m.centers, m.decays, "truncated")
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    ks = stats.kstest(s, lambda x: np.interp(x, grid, cdf))
    assert ks.statistic < 0.01


def test_fleury_spread_limits():
    assert D.fleury_angular_spread(np.zeros(5)) == pytest.approx(0.0, abs=1e-12)
    phi = np.linspace(-np.pi, np.pi, 3600, endpoint=False)
    assert D.fleury_angular_spread(phi) == pytest.approx(1.0, abs=1e-9)
    # two equal paths at +-a: spread = sin(a)
    a = 0.3
    assert D.fleury_angular_spread(np.array([-a, a])) == pytest.approx(np.sin(a), abs=1e-12)


def test_fleury_is_rotation_invariant():
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.3, 50)
    w = rng.random(50)
    s0 = D.fleury_angular_spread(phi, w)
    assert D.fleury_angular_spread(D.wrap_angle(phi + 2.7), w) == pytest.approx(s0, abs=1e-12)


def test_rms_delay_spread_two_taps():
    # two equal taps separated by d -> spread d/2
    assert D.rms_delay_spread(np.array([0.0, 4.0]), np.array([1.0, 1.0])) == pytest.approx(2.0)
    assert D.mean_delay(np.array([0.0, 4.0]), np.array([3.0, 1.0])) == pytest.approx(1.0)


def test_ecdf():
    x, f = D.ecdf([3, 1, 2])
    assert x.tolist() == [1, 2, 3]
    assert np.allclose(f, [1 / 3, 2 / 3, 1])
