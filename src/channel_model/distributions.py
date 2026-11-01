"""Probability distributions and circular statistics used by the model (Sec. V).

* Gamma (cluster number, arrival time) with the paper's (shape a, scale b) form
* Log-normal (intra-cluster spreads):  ln X ~ N(mu, sigma^2)
* Bimodal Laplace on the circle (Eq. 3) and the uniform circular law
* Power-weighted circular mean (Eq. 2) and Fleury's RMS angular spread (Eq. 4 / [56])
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- helpers
def wrap_angle(x: ArrayLike) -> NDArray[np.float64]:
    """Map angles to [-pi, pi) (the modulo operation written as a hat in Eq. 3)."""
    return (np.asarray(x, dtype=float) + np.pi) % TWO_PI - np.pi


# --------------------------------------------------------------------------- Gamma
def gamma_sample(shape: float, scale: float, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray:
    return rng.gamma(shape, scale, size=size)


def gamma_pdf(x: ArrayLike, shape: float, scale: float) -> NDArray:
    return stats.gamma.pdf(x, a=shape, scale=scale)


def gamma_cdf(x: ArrayLike, shape: float, scale: float) -> NDArray:
    return stats.gamma.cdf(x, a=shape, scale=scale)


def round_cluster_count(x: ArrayLike) -> NDArray[np.int64]:
    """Footnote 3: Gamma draws are rounded to the nearest integer (half away from zero)."""
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(np.int64)


# --------------------------------------------------------------------------- log-normal
def lognormal_sample(mu: float, sigma: float, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray:
    """X = exp(mu + sigma Z): mu/sigma are moments of ln X, not of X."""
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def lognormal_pdf(x: ArrayLike, mu: float, sigma: float) -> NDArray:
    return stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))


def lognormal_cdf(x: ArrayLike, mu: float, sigma: float) -> NDArray:
    return stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))


# --------------------------------------------------------------------------- bimodal Laplace (Eq. 3)
def bimodal_laplace_pdf(phi: ArrayLike, weights, centers, decays, mode: str = "truncated") -> NDArray:
    """Density of the cluster-centroid angle on [-pi, pi).

    ``mode='truncated'``: Eq. (3) as written, sum_i w_i/(2 lam_i) exp(-|wrap(phi-mu_i)|/lam_i),
    renormalised so that it integrates to one over the circle (Eq. 3 itself has mass
    sum_i w_i (1 - exp(-pi/lam_i)) < 1 for wide components).
    ``mode='wrapped'``: the wrapped Laplace mixture (sum over all 2*pi images).
    """
    phi = np.asarray(phi, dtype=float)
    w, mu, lam = (np.asarray(v, dtype=float) for v in (weights, centers, decays))
    out = np.zeros_like(phi, dtype=float)
    if mode == "truncated":
        mass = 0.0
        for wi, mi, li in zip(w, mu, lam):
            out += wi / (2 * li) * np.exp(-np.abs(wrap_angle(phi - mi)) / li)
            mass += wi * (1.0 - np.exp(-np.pi / li))
        return out / mass
    if mode == "wrapped":
        for wi, mi, li in zip(w, mu, lam):
            d = np.abs(wrap_angle(phi - mi))
            # closed form of sum_k exp(-|d + 2 pi k|/l) for d in [0, pi]
            q = np.exp(-TWO_PI / li)
            out += wi / (2 * li) * (np.exp(-d / li) + np.exp(-(TWO_PI - d) / li)) / (1.0 - q)
        return out
    raise ValueError("mode must be 'truncated' or 'wrapped'")


def bimodal_laplace_sample(weights, centers, decays, size: int, rng: np.random.Generator,
                           mode: str = "truncated") -> NDArray:
    """Draw centroid angles from the bimodal Laplace model (Eq. 3)."""
    w, mu, lam = (np.asarray(v, dtype=float) for v in (weights, centers, decays))
    if mode == "truncated":
        # component probabilities proportional to the mass each term has on the circle
        p = w * (1.0 - np.exp(-np.pi / lam))
        p = p / p.sum()
        comp = rng.choice(len(w), size=size, p=p)
        l = lam[comp]
        # |offset| ~ exponential(l) truncated to [0, pi]: inverse-CDF sampling
        u = rng.random(size)
        mag = -l * np.log1p(-u * (1.0 - np.exp(-np.pi / l)))
        sign = np.where(rng.random(size) < 0.5, -1.0, 1.0)
        return wrap_angle(mu[comp] + sign * mag)
    if mode == "wrapped":
        comp = rng.choice(len(w), size=size, p=w / w.sum())
        return wrap_angle(mu[comp] + rng.laplace(0.0, lam[comp], size=size))
    raise ValueError("mode must be 'truncated' or 'wrapped'")


def uniform_circular_sample(size: int, rng: np.random.Generator) -> NDArray:
    """NLoS centroids: uniform on [0, 2 pi), returned in the [-pi, pi) convention."""
    return wrap_angle(rng.uniform(0.0, TWO_PI, size=size))


# --------------------------------------------------------------------------- circular statistics
def circular_mean(phi: ArrayLike, weights: ArrayLike | None = None, axis: int | None = None) -> NDArray:
    """Power-weighted centroid angle, Eq. (2):  angle( sum p e^{j phi} / sum p )."""
    phi = np.asarray(phi, dtype=float)
    w = np.ones_like(phi) if weights is None else np.asarray(weights, dtype=float)
    z = np.sum(w * np.exp(1j * phi), axis=axis) / np.sum(w, axis=axis)
    return np.angle(z)


def mean_resultant(phi: ArrayLike, weights: ArrayLike | None = None, axis: int | None = None) -> NDArray:
    """Complex power-weighted mean phasor mu = sum p e^{j phi} / sum p (Fleury [56])."""
    phi = np.asarray(phi, dtype=float)
    w = np.ones_like(phi) if weights is None else np.asarray(weights, dtype=float)
    return np.sum(w * np.exp(1j * phi), axis=axis) / np.sum(w, axis=axis)


def fleury_angular_spread(phi: ArrayLike, weights: ArrayLike | None = None, axis: int | None = None) -> NDArray:
    """RMS angular spread of Fleury [56] used by the paper (Eq. 4) and by [21] (Eq. 9).

    sigma = sqrt( sum p |e^{j phi} - mu|^2 / sum p ) = sqrt(1 - |mu|^2),
    unitless and bounded by 1.  See docs/PAPER_ANALYSIS.md for why the square root is
    used even though it is missing in the typeset Eq. (4).
    """
    mu = mean_resultant(phi, weights, axis=axis)
    return np.sqrt(np.clip(1.0 - np.abs(mu) ** 2, 0.0, 1.0))


def eq4_as_typeset(phi: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Literal transcription of the printed Eq. (4) (no square root, unit-phasor reference).

    Only kept for documentation/tests: it can reach 2, contradicting the stated maximum of 1.
    """
    phi = np.asarray(phi, dtype=float)
    w = np.ones_like(phi) if weights is None else np.asarray(weights, dtype=float)
    ref = np.exp(1j * circular_mean(phi, w))
    return float(np.sum(np.abs(np.exp(1j * phi) - ref) ** 2 * w) / np.sum(w))


def rms_delay_spread(tau: ArrayLike, power: ArrayLike, axis: int | None = None) -> NDArray:
    """Square root of the second central moment of a (cluster or channel) delay profile."""
    tau = np.asarray(tau, dtype=float)
    p = np.asarray(power, dtype=float)
    s = np.sum(p, axis=axis)
    m1 = np.sum(p * tau, axis=axis) / s
    m2 = np.sum(p * tau**2, axis=axis) / s
    return np.sqrt(np.clip(m2 - m1**2, 0.0, None))


def mean_delay(tau: ArrayLike, power: ArrayLike, axis: int | None = None) -> NDArray:
    tau = np.asarray(tau, dtype=float)
    p = np.asarray(power, dtype=float)
    return np.sum(p * tau, axis=axis) / np.sum(p, axis=axis)


def ecdf(x: ArrayLike) -> tuple[NDArray, NDArray]:
    """Empirical CDF as (sorted values, probabilities i/n)."""
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    return xs, np.arange(1, xs.size + 1) / xs.size
