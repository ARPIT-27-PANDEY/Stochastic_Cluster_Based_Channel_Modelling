"""Dynamic-range (power cut-off) modelling of cluster statistics (Sec. VI).

A cluster with peak power normalised to one has the ADPS shape (Eqs. 12-13)

    p(tau, phiR, phiT) = exp(-tau/k_tau - |phiR|/k_R - |phiT|/k_T),
    tau >= 0, phiR, phiT in [-pi, pi).

With an effective dynamic range DR_eff (dB) relative to the cluster peak, a tap is
observable iff  tau/k_tau + |phiR|/k_R + |phiT|/k_T <= L,  L = DR_eff ln(10)/10.
The cut-off therefore couples the three domains (the integration region is a
truncated simplex), which is why the paper says no closed-form inverse exists.

This module provides
* :func:`continuous_observables`  - S_obs = F(kappa; DR_eff) on the continuum.  The
  delay integral and one angular integral are solved in closed form; the last angular
  integral uses piecewise Gauss-Legendre quadrature.  All operations are analytic in
  the parameters, so complex-step differentiation gives an exact Jacobian.
  Eq. (14) of the paper is the special case without angular truncation.
* :func:`discrete_observables`    - the same quantities on a finite delay/angle grid
  (what a sounder with 0.3 m / 10 deg bins observes), used for the discretisation
  correction matrix and for validating generated clusters.
* :func:`regime_boundaries_db`    - the dynamic-range values that separate the four
  observable regimes discussed with Figs. 14-15.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

LN10_10 = np.log(10.0) / 10.0
_GL_T, _GL_W = np.polynomial.legendre.leggauss(64)


def db_to_nepers_power(dr_db: ArrayLike) -> NDArray:
    """L = DR ln(10)/10: the cut-off expressed in the exponent of the power profile."""
    return np.asarray(dr_db) * LN10_10


def _rwhere(cond_real, a, b):
    return np.where(cond_real, a, b)


def _G(n: int, c, V):
    """Closed form of  int_0^V e^{-v} gamma(n+1, c - v) dv  for 0 <= V <= c (gamma = lower incomplete)."""
    fact = (1.0, 1.0, 2.0)[n]
    s = 0.0
    kf = 1.0
    for k in range(n + 1):
        kf *= (k + 1)  # (k+1)!
        s = s + (c ** (k + 1) - (c - V) ** (k + 1)) / kf
    return fact * (1.0 - np.exp(-V)) - fact * np.exp(-c) * s


def _outer_integrals(a_out, lim_out, lim_in, L, moments: bool = True):
    """Return (M0, M1, M2, C) integrated over the 'outer' angle u in [0, min(lim_out, L)].

    M_n = int e^{-u} G_n(L-u, min(lim_in, L-u)) du ;  C = int e^{-u} cos(a_out u) G_0(...) du.
    u is the outer angle in units of its decay constant (u = |phi|/a_out).
    """
    U = _rwhere(np.real(lim_out) < np.real(L), lim_out, L)
    uk = L - lim_in  # kink where the inner limit switches from lim_in to L-u
    s = _rwhere(np.real(uk) <= 0, 0.0 * U, _rwhere(np.real(uk) >= np.real(U), U, uk))
    t = _GL_T[None, :]
    w = _GL_W[None, :]
    out = [0.0, 0.0, 0.0, 0.0]
    for lo, hi, inner_fixed in ((0.0 * s, s, True), (s, U, False)):
        lo_ = lo[:, None]
        hi_ = hi[:, None]
        u = lo_ + (hi_ - lo_) * (t + 1.0) / 2.0
        jac = (hi_ - lo_) / 2.0
        c = L[:, None] - u
        V = (lim_in[:, None] + 0.0 * u) if inner_fixed else c
        eu = np.exp(-u)
        wg = w * jac * eu
        g0 = _G(0, c, V)
        if moments:
            out[0] = out[0] + np.sum(wg * g0, axis=1)
            out[1] = out[1] + np.sum(wg * _G(1, c, V), axis=1)
            out[2] = out[2] + np.sum(wg * _G(2, c, V), axis=1)
        out[3] = out[3] + np.sum(wg * np.cos(a_out[:, None] * u) * g0, axis=1)
    return out


def continuous_observables(kappa_tau: ArrayLike, kappa_r: ArrayLike, kappa_t: ArrayLike,
                           dr_db: ArrayLike) -> dict[str, NDArray]:
    """Observable cluster statistics under a joint dynamic-range cut-off (continuum).

    Parameters may be real or complex (complex-step differentiation).  Arrays broadcast.
    Returns mean delay, RMS delay spread (same unit as ``kappa_tau``), and the Fleury
    DoA / DoD angular spreads.  For DR -> inf: mean = spread = kappa_tau, and the angular
    spreads tend to those of a Laplacian truncated to [-pi, pi).
    """
    kt, a, b, dr = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v)) for v in
                                         (kappa_tau, kappa_r, kappa_t, dr_db)))
    shape = kt.shape
    kt, a, b, dr = (v.ravel() for v in (kt, a, b, dr))
    L = dr * LN10_10
    Ur, Vt = np.pi / a, np.pi / b
    m0, m1, m2, cr = _outer_integrals(a, Ur, Vt, L)
    _, _, _, ct = _outer_integrals(b, Vt, Ur, L, moments=False)
    mean_n = m1 / m0
    var_n = m2 / m0 - mean_n**2
    mu_r = cr / m0
    mu_t = ct / m0
    res = {
        "mean_tau": (kt * mean_n).reshape(shape),
        "s_tau": (kt * np.sqrt(var_n)).reshape(shape),
        "s_r": _sqrt1m(mu_r).reshape(shape),
        "s_t": _sqrt1m(mu_t).reshape(shape),
        "delay_ratio": np.sqrt(var_n).reshape(shape),
    }
    return res


def _sqrt1m(mu):
    """sqrt(1 - mu^2), clipped at 0 for real input (round-off), analytic for complex-step input."""
    v = 1.0 - mu**2
    if not np.iscomplexobj(v):
        v = np.clip(v, 0.0, None)
    return np.sqrt(v)


def eq14_delay_spread(kappa_tau: ArrayLike, tau_tr: ArrayLike) -> NDArray:
    """Eq. (14): RMS spread of a one-sided exponential truncated at tau_tr (delay-only cut-off)."""
    k = np.asarray(kappa_tau, float)
    x = np.asarray(tau_tr, float) / k
    e = np.exp(-x)
    m0 = 1 - e
    m1 = 1 - e * (1 + x)
    m2 = 2 - e * (x**2 + 2 * x + 2)
    return k * np.sqrt(m2 / m0 - (m1 / m0) ** 2)


def infinite_dr_angular_spread(kappa: ArrayLike) -> NDArray:
    """Fleury spread of exp(-|phi|/kappa) on [-pi, pi) (the DR -> infinity limit)."""
    k = np.asarray(kappa, float)
    # int_0^pi e^{-p/k} cos p dp / int_0^pi e^{-p/k} dp
    num = (k / (1 + k**2)) * (1 + np.exp(-np.pi / k))
    den = k * (1 - np.exp(-np.pi / k))
    mu = num / den
    return np.sqrt(1 - mu**2)


# --------------------------------------------------------------------------- discrete grid
@dataclass(frozen=True)
class AngleGrid:
    step_deg: float = 10.0

    @property
    def n(self) -> int:
        return int(round(360.0 / self.step_deg))

    @property
    def phi(self) -> NDArray:
        """Bin centres in [-pi, pi), with 0 (boresight) on the grid."""
        k = np.arange(self.n) - self.n // 2
        return np.deg2rad(k * self.step_deg)


def _geom_sums(q, K):
    """sum_{k=0}^{K} k^n q^k for n = 0, 1, 2 (closed forms; K >= 0 integer array)."""
    omq = 1.0 - q
    qK = q**K
    s0 = (1.0 - qK * q) / omq
    s1 = q * (1.0 - (K + 1) * qK + K * qK * q) / omq**2
    s2 = q * (1.0 + q - (K + 1) ** 2 * qK + (2 * K**2 + 2 * K - 1) * qK * q - K**2 * qK * q * q) / omq**3
    return s0, s1, s2


def discrete_observables(kappa_tau_bins: ArrayLike, kappa_r: ArrayLike, kappa_t: ArrayLike, dr_db: ArrayLike,
                         angle_grid: AngleGrid = AngleGrid(), delay_res: float = 1.0,
                         max_delay_bins: int | None = None) -> dict[str, NDArray]:
    """Observable statistics of a grid-aligned cluster after the DR cut-off.

    ``kappa_tau_bins`` is the delay decay constant in units of delay bins.  The cluster
    origin sits on a grid point (delay bin 0, angle bin 0).  Returns delay mean/spread in
    the units of ``delay_res`` (pass the bin size in metres to obtain metres), Fleury
    angular spreads, the number of surviving taps and the sum of the surviving
    normalised tap powers (needed to scale a cluster to its observable power).
    """
    kb, a, b, dr = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in
                                         (kappa_tau_bins, kappa_r, kappa_t, dr_db)))
    shape = kb.shape
    kb, a, b, dr = (v.ravel() for v in (kb, a, b, dr))
    L = dr * LN10_10
    phi = angle_grid.phi
    u = np.abs(phi)[None, :] / a[:, None]            # (N, nA)
    v = np.abs(phi)[None, :] / b[:, None]
    r = L[:, None, None] - u[:, :, None] - v[:, None, :]  # (N, nR, nT)
    alive = r >= -1e-12
    K = np.floor(np.clip(r, 0.0, None) * kb[:, None, None] + 1e-9)
    if max_delay_bins is not None:
        K = np.minimum(K, max_delay_bins - 1)
    q = np.exp(-1.0 / kb)[:, None, None]
    s0, s1, s2 = _geom_sums(q, K)
    wa = np.exp(-u[:, :, None] - v[:, None, :]) * alive
    W0, W1, W2 = wa * s0, wa * s1, wa * s2
    m0 = W0.sum(axis=(1, 2))
    m1 = W1.sum(axis=(1, 2))
    m2 = W2.sum(axis=(1, 2))
    mean = m1 / m0
    var = np.clip(m2 / m0 - mean**2, 0.0, None)
    aps_r = W0.sum(axis=2)
    aps_t = W0.sum(axis=1)
    e = np.exp(1j * phi)[None, :]
    mu_r = np.abs((aps_r * e).sum(1) / aps_r.sum(1))
    mu_t = np.abs((aps_t * e).sum(1) / aps_t.sum(1))
    ntaps = ((K + 1) * alive).sum(axis=(1, 2))
    return {
        "mean_tau": (mean * delay_res).reshape(shape),
        "s_tau": (np.sqrt(var) * delay_res).reshape(shape),
        "s_r": np.sqrt(np.clip(1 - mu_r**2, 0, 1)).reshape(shape),
        "s_t": np.sqrt(np.clip(1 - mu_t**2, 0, 1)).reshape(shape),
        "shape_sum": m0.reshape(shape),
        "n_taps": ntaps.reshape(shape),
    }


def regime_boundaries_db(kappa_r: float, kappa_t: float) -> tuple[float, float, float]:
    """Dynamic ranges (dB) separating the four observable regimes of Figs. 14-15.

    An angular dimension saturates when the cut-off allows its full support [-pi, pi):
    L >= pi/kappa.  Regimes: (i) both angles DR-limited, (ii) the wider cluster dimension
    saturated, (iii) the second one saturated too, but the corner (both at +-pi) not yet
    reached, (iv) L >= pi/k_R + pi/k_T: the complete angular square is inside the cut-off.
    """
    l1 = np.pi / max(kappa_r, kappa_t)
    l2 = np.pi / min(kappa_r, kappa_t)
    l3 = np.pi / kappa_r + np.pi / kappa_t
    return tuple(float(x / LN10_10) for x in (l1, l2, l3))


def effective_dynamic_range(peak_db: ArrayLike, cutoff_db: float) -> NDArray:
    """DR_eff of each cluster = its peak (dB) minus the global cut-off level (dB)."""
    return np.asarray(peak_db, float) - cutoff_db
