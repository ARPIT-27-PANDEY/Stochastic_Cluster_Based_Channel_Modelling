"""Cluster power model (Sec. V-F, Eqs. 5-11).

* free-space path loss and normalised cluster power (NCP, Eq. 6)
* distance-ratio dependent Gaussian NCP model (Eqs. 7-9)
* distance-equalisation weights (Eq. 10)
* truncated-normal log-likelihood (Eq. 11) and its maximisation by alternating
  fixed-step gradient ascent with multi-start (paper's procedure), plus a
  quasi-Newton cross-check and the ordinary-least-squares reference fit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special, stats

from .parameters import PowerParams

C0 = 299792458.0


def fspl_db(d_m: ArrayLike, fc_hz: float, c: float = C0) -> NDArray:
    """Friis free-space path loss 20 log10(4 pi d f / c) in dB (isotropic antennas)."""
    d = np.asarray(d_m, dtype=float)
    if np.any(d <= 0):
        raise ValueError("distance must be positive")
    return 20.0 * np.log10(4.0 * np.pi * d * fc_hz / c)


def distance_ratio_db(d_m: ArrayLike, d_los_m: ArrayLike) -> NDArray:
    """x = 10 log10(d / d_LoS), the regressor of Eqs. (8)-(9)."""
    return 10.0 * np.log10(np.asarray(d_m, dtype=float) / np.asarray(d_los_m, dtype=float))


def ncp_mean(x_db: ArrayLike, p: PowerParams) -> NDArray:
    return p.alpha + p.beta * np.asarray(x_db, dtype=float)


def ncp_std(x_db: ArrayLike, p: PowerParams, min_sigma: float = 0.1) -> NDArray:
    return np.maximum(p.rho + p.eta * np.asarray(x_db, dtype=float), min_sigma)


def sample_ncp(x_db: ArrayLike, p: PowerParams, rng: np.random.Generator, min_sigma: float = 0.1) -> NDArray:
    """Draw NCP[dB] = P_bar + chi_sigma  (Eq. 8)."""
    x = np.asarray(x_db, dtype=float)
    return ncp_mean(x, p) + ncp_std(x, p, min_sigma) * rng.standard_normal(x.shape)


def ncp_to_cluster_power_db(ncp_db: ArrayLike, d_los_m: float, fc_hz: float) -> NDArray:
    """Invert Eq. (6):  [P_c]dB = NCP - FSPL(d_LoS)."""
    return np.asarray(ncp_db, dtype=float) - fspl_db(d_los_m, fc_hz)


# --------------------------------------------------------------------------- Eq. (10)
def equalization_weights(x_db: ArrayLike, n_bins: int = 20, x_range: tuple[float, float] = (0.0, 20.0),
                         min_fraction: float = 0.01) -> NDArray:
    """Distance-equalisation weights of Eq. (10).

    Points in bin i get w_i = (1/N_b) N_s/N_i if N_i/N_s >= min_fraction, else 1.
    """
    x = np.asarray(x_db, dtype=float)
    ns = x.size
    edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    frac = counts / ns
    wbin = np.where(frac >= min_fraction, (1.0 / n_bins) * ns / np.maximum(counts, 1.0), 1.0)
    return wbin[idx]


# --------------------------------------------------------------------------- Eq. (11)
def _inv_mills(a: NDArray) -> NDArray:
    """phi(a) / Phi(a), numerically stable for very negative a."""
    return np.exp(stats.norm.logpdf(a) - special.log_ndtr(a))


def truncated_llf(theta: ArrayLike, x: NDArray, y: NDArray, thr: NDArray, w: NDArray,
                  min_sigma: float = 1e-3) -> float:
    """Weighted log-likelihood of observed NCPs under truncation at ``thr`` (Eq. 11).

    Written in the power domain: an observation exists only if y >= thr, so the density
    is N(y; mu, sigma^2) / P(Y >= thr) with P(Y >= thr) = Phi((mu - thr)/sigma).
    This is algebraically identical to the paper's path-loss-domain form
    -ln Phi((PL_thr - PL_bar)/sigma), because PL = -power.
    """
    a_, b_, r_, e_ = theta
    mu = a_ + b_ * x
    sg = r_ + e_ * x
    if np.any(sg <= min_sigma):
        return -np.inf
    z = (y - mu) / sg
    return float(np.sum(w * (-np.log(sg) - 0.5 * z**2 - special.log_ndtr((mu - thr) / sg))))


def truncated_llf_grad(theta: ArrayLike, x: NDArray, y: NDArray, thr: NDArray, w: NDArray) -> NDArray:
    """Analytic gradient of :func:`truncated_llf` w.r.t. (alpha, beta, rho, eta)."""
    a_, b_, r_, e_ = theta
    mu = a_ + b_ * x
    sg = r_ + e_ * x
    a = (mu - thr) / sg
    lam = _inv_mills(a)
    d_mu = w * ((y - mu) / sg**2 - lam / sg)
    d_sg = w * (-1.0 / sg + (y - mu) ** 2 / sg**3 + lam * (mu - thr) / sg**2)
    return np.array([d_mu.sum(), (d_mu * x).sum(), d_sg.sum(), (d_sg * x).sum()])


@dataclass
class MLEResult:
    alpha: float
    beta: float
    rho: float
    eta: float
    llf: float
    iterations: int
    converged: bool
    n_starts: int
    history: list[float]

    def as_params(self) -> PowerParams:
        return PowerParams(self.alpha, self.beta, self.rho, self.eta)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("history")
        return d


def _alternating_ascent(theta0: NDArray, x, y, thr, w, lr: float, max_iter: int, tol: float,
                        record_every: int = 500) -> tuple[NDArray, float, int, bool, list[float]]:
    """Alternating fixed-step gradient ascent (Sec. V-F-2).

    One sweep = step on (alpha, beta) with (rho, eta) fixed, then step on (rho, eta).
    The step is the fixed learning rate times the gradient.  Safeguard (our addition,
    documented): if a step would make sigma non-positive or lower the LLF, the step for
    that block is halved until it does not (the nominal rate is restored next sweep).
    """
    theta = theta0.astype(float).copy()
    llf = truncated_llf(theta, x, y, thr, w)
    hist = [llf]
    for it in range(1, max_iter + 1):
        llf_old = llf
        for block in (slice(0, 2), slice(2, 4)):
            g = truncated_llf_grad(theta, x, y, thr, w)
            step = lr
            for _ in range(30):
                cand = theta.copy()
                cand[block] += step * g[block]
                llf_c = truncated_llf(cand, x, y, thr, w)
                if np.isfinite(llf_c) and llf_c >= llf - 1e-12:
                    theta, llf = cand, llf_c
                    break
                step *= 0.5
        if it % record_every == 0:
            hist.append(llf)
        if abs(llf - llf_old) <= tol * max(1.0, abs(llf)):
            hist.append(llf)
            return theta, llf, it, True, hist
    return theta, llf, max_iter, False, hist


def fit_truncated_mle(x_db: ArrayLike, ncp_db: ArrayLike, thr_db: ArrayLike, weights: ArrayLike | None = None,
                      lr: float = 1e-3, max_iter: int = 200000, tol: float = 1e-10, n_starts: int = 8,
                      start_box: dict | None = None, rng: np.random.Generator | None = None) -> MLEResult:
    """Multi-start alternating gradient-ascent MLE of (alpha, beta, rho, eta), Sec. V-F-2."""
    x = np.asarray(x_db, float)
    y = np.asarray(ncp_db, float)
    thr = np.broadcast_to(np.asarray(thr_db, float), x.shape)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    if np.any(y < thr - 1e-9):
        raise ValueError("observed NCPs must lie above their truncation thresholds")
    rng = rng or np.random.default_rng(0)
    box = start_box or {"alpha": [-50, 10], "beta": [-4, 1], "rho": [2, 15], "eta": [-0.5, 0.2]}
    best: MLEResult | None = None
    xmax = max(float(x.max()), 1e-6)
    for s in range(n_starts):
        if s == 0:  # deterministic start from the OLS mean and pooled std
            ols = fit_ols(x, y)
            th0 = np.array([ols[0], ols[1], max(np.std(y), 1.0), 0.0])
        else:
            th0 = np.array([rng.uniform(*box[k]) for k in ("alpha", "beta", "rho", "eta")])
            # make sure the start is feasible (sigma > 0 on the data range)
            th0[3] = max(th0[3], -(th0[2] - 0.5) / xmax)
        th, llf, it, conv, hist = _alternating_ascent(th0, x, y, thr, w, lr, max_iter, tol)
        res = MLEResult(*th, llf=llf, iterations=it, converged=conv, n_starts=n_starts, history=hist)
        if best is None or res.llf > best.llf:
            best = res
    assert best is not None
    return best


def fit_truncated_mle_lbfgs(x_db, ncp_db, thr_db, weights=None, theta0=None) -> tuple[NDArray, float]:
    """Cross-check of the gradient-ascent result with L-BFGS-B (verification only)."""
    x = np.asarray(x_db, float)
    y = np.asarray(ncp_db, float)
    thr = np.broadcast_to(np.asarray(thr_db, float), x.shape)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    if theta0 is None:
        ols = fit_ols(x, y)
        theta0 = np.array([ols[0], ols[1], np.std(y), 0.0])
    f = lambda th: -truncated_llf(th, x, y, thr, w) if np.isfinite(truncated_llf(th, x, y, thr, w)) else 1e30
    g = lambda th: -truncated_llf_grad(th, x, y, thr, w)
    r = optimize.minimize(f, theta0, jac=g, method="L-BFGS-B")
    return r.x, -r.fun


def fit_ols(x_db: ArrayLike, ncp_db: ArrayLike, weights: ArrayLike | None = None) -> tuple[float, float]:
    """Least-squares mean fit that ignores truncation (red dot-dashed line of Figs. 12-13)."""
    x = np.asarray(x_db, float)
    y = np.asarray(ncp_db, float)
    w = None if weights is None else np.sqrt(np.asarray(weights, float))
    beta, alpha = np.polyfit(x, y, 1, w=w)
    return float(alpha), float(beta)


def sigma_coverage(x_db, ncp_db, thr_db, p: PowerParams, k: float = 1.0) -> tuple[float, float]:
    """(empirical, truncated-Gaussian expected) share of observations within mu +- k sigma."""
    x = np.asarray(x_db, float)
    y = np.asarray(ncp_db, float)
    thr = np.broadcast_to(np.asarray(thr_db, float), x.shape)
    mu = ncp_mean(x, p)
    sg = ncp_std(x, p)
    emp = float(np.mean(np.abs(y - mu) <= k * sg))
    t = (thr - mu) / sg
    num = np.clip(stats.norm.cdf(k) - stats.norm.cdf(np.maximum(-k, t)), 0.0, None)
    den = stats.norm.sf(t)
    exp_ = float(np.mean(num / np.maximum(den, 1e-300)))
    return emp, exp_
