"""Inverse mapping kappa = F^{-1}(S_obs; DR_eff) and the discretisation correction (Sec. VI).

* :func:`invert_spreads` - vectorised Newton-Raphson on log(kappa) with the exact
  Jacobian of the truncation model obtained by complex-step differentiation of
  :func:`dynamic_range.continuous_observables` (the paper evaluates the Jacobian
  "analytically from the truncation model"; complex-step gives the same derivative to
  machine precision without hand-deriving every term).
* :class:`DiscretizationCorrection` - the offline correction matrix: for a grid of
  decay constants and dynamic ranges it stores Delta = S_discrete - S_continuous for the
  native delay/angle grid.  During generation the drawn (grid-observable) spreads are
  compensated, S_cont = S_obs - Delta(kappa, DR), before the inverse mapping; the two
  steps are iterated a few times because Delta is tabulated against kappa.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

from .dynamic_range import AngleGrid, continuous_observables, discrete_observables

_KEYS = ("s_tau", "s_r", "s_t")
LOG_KAPPA_MIN = np.log(1e-5)
LOG_KAPPA_MAX = np.log(1e4)


def _forward(logk: NDArray, dr: NDArray) -> NDArray:
    r = continuous_observables(np.exp(logk[:, 0]), np.exp(logk[:, 1]), np.exp(logk[:, 2]), dr)
    return np.stack([r[k] for k in _KEYS], axis=1)


@dataclass
class InverseResult:
    kappa: NDArray          # (N, 3): kappa_tau [same unit as S_tau], kappa_R, kappa_T [rad]
    converged: NDArray      # (N,) bool
    iterations: int
    residual: NDArray       # (N,) max relative residual


def _angular(la: NDArray, lb: NDArray, dr: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    r = continuous_observables(1.0, np.exp(la), np.exp(lb), dr)
    return r["s_r"], r["s_t"], r["delay_ratio"]


def _bisect_1d(target: NDArray, dr: NDArray, which: int, n_iter: int = 40) -> NDArray:
    """Robust start value: solve S_phi(kappa) = target with the *other* angle very narrow."""
    lo = np.full(target.shape, LOG_KAPPA_MIN)
    hi = np.full(target.shape, LOG_KAPPA_MAX)
    narrow = np.full(target.shape, np.log(1e-4))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        s_r, s_t, _ = _angular(mid, narrow, dr) if which == 0 else _angular(narrow, mid, dr)
        val = s_r if which == 0 else s_t
        up = val < target            # spread increases monotonically with kappa
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def invert_spreads(s_obs: ArrayLike, dr_db: ArrayLike, max_iter: int = 40, tol: float = 1e-10,
                   kappa0: ArrayLike | None = None) -> InverseResult:
    """Solve F(kappa; DR) = S_obs for kappa = [k_tau, k_R, k_T] (vectorised over clusters).

    Newton-Raphson with the exact Jacobian (complex step).  Because the delay axis is
    scale-free, S_R and S_T do not depend on k_tau and S_tau = k_tau g(DR, k_R, k_T); the
    3x3 Jacobian is therefore block-triangular and the Newton system is solved as a 2x2
    angular system followed by the exact delay update k_tau = S_tau / g.  Starting values
    come from ``kappa0`` (warm start) or from 1-D bisection in each angular domain; the
    better of the two (smaller residual) is used per cluster.

    ``s_obs``: (N, 3) array [S_tau, S_DoA, S_DoD]; angular spreads must be in (0, 1).
    """
    s = np.atleast_2d(np.asarray(s_obs, float)).copy()
    n = s.shape[0]
    dr = np.broadcast_to(np.asarray(dr_db, float), (n,)).astype(float)
    if np.any(dr <= 0):
        raise ValueError("effective dynamic range must be positive")
    if np.any(s[:, 0] <= 0):
        raise ValueError("delay spread must be positive")
    s[:, 1:] = np.clip(s[:, 1:], 1e-6, 1 - 1e-9)
    tgt = s[:, 1:]
    la = np.zeros(n)
    lb = np.zeros(n)
    err = np.full(n, np.inf)
    if kappa0 is not None:
        k0 = np.atleast_2d(np.asarray(kappa0, float))
        ok0 = np.all(np.isfinite(k0) & (k0 > 0), axis=1)
        if ok0.any():
            la[ok0] = np.log(k0[ok0, 1])
            lb[ok0] = np.log(k0[ok0, 2])
            sr0, st0, _ = _angular(la[ok0], lb[ok0], dr[ok0])
            err[ok0] = np.maximum(np.abs(sr0 / tgt[ok0, 0] - 1), np.abs(st0 / tgt[ok0, 1] - 1))
    need = ~(err < 1e-2)   # bisection start only where no good warm start exists
    if need.any():
        ba = _bisect_1d(tgt[need, 0], dr[need], 0, 30)
        bb = _bisect_1d(tgt[need, 1], dr[need], 1, 30)
        sr, st, _ = _angular(ba, bb, dr[need])
        eb = np.maximum(np.abs(sr / tgt[need, 0] - 1), np.abs(st / tgt[need, 1] - 1))
        better = eb < err[need]
        j = np.nonzero(need)[0][better]
        la[j], lb[j], err[j] = ba[better], bb[better], eb[better]
    h = 1e-30
    it = 0
    for it in range(1, max_iter + 1):
        active = err > tol
        if not active.any():
            break
        i = np.nonzero(active)[0]
        a_, b_, d_ = la[i], lb[i], dr[i]
        f_r, f_t, _ = _angular(a_, b_, d_)
        J = np.empty((i.size, 2, 2))
        r1, t1, _ = _angular(a_ + 1j * h, b_.astype(complex), d_)
        r2, t2, _ = _angular(a_.astype(complex), b_ + 1j * h, d_)
        J[:, 0, 0], J[:, 1, 0] = r1.imag / h, t1.imag / h
        J[:, 0, 1], J[:, 1, 1] = r2.imag / h, t2.imag / h
        rhs = np.column_stack([tgt[i, 0] - f_r, tgt[i, 1] - f_t])
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        step = np.column_stack([(J[:, 1, 1] * rhs[:, 0] - J[:, 0, 1] * rhs[:, 1]) / det,
                                (J[:, 0, 0] * rhs[:, 1] - J[:, 1, 0] * rhs[:, 0]) / det])
        step = np.clip(np.nan_to_num(step), -1.5, 1.5)
        lam = np.ones(i.size)
        pending = np.ones(i.size, bool)
        for _ in range(10):
            j = np.nonzero(pending)[0]
            ca = np.clip(a_[j] + lam[j] * step[j, 0], LOG_KAPPA_MIN, LOG_KAPPA_MAX)
            cb = np.clip(b_[j] + lam[j] * step[j, 1], LOG_KAPPA_MIN, LOG_KAPPA_MAX)
            cr, ct, _ = _angular(ca, cb, d_[j])
            ce = np.maximum(np.abs(cr / tgt[i[j], 0] - 1), np.abs(ct / tgt[i[j], 1] - 1))
            ok = np.isfinite(ce) & (ce < err[i[j]])
            acc = j[ok]
            la[i[acc]], lb[i[acc]], err[i[acc]] = ca[ok], cb[ok], ce[ok]
            pending[acc] = False
            if not pending.any():
                break
            lam[pending] *= 0.5
        if pending.all():
            break  # no cluster improved: stagnation (residual reported)
    _, _, g = _angular(la, lb, dr)
    kt = s[:, 0] / g
    kappa = np.column_stack([kt, np.exp(la), np.exp(lb)])
    return InverseResult(kappa=kappa, converged=err <= max(tol, 1e-7), iterations=it, residual=err)


def invert_spreads_full3d(s_obs: ArrayLike, dr_db: ArrayLike, max_iter: int = 60, tol: float = 1e-10,
                          kappa0: ArrayLike | None = None) -> InverseResult:
    """Plain 3-D Newton-Raphson on log(kappa) (reference implementation used in the tests).

    ``s_obs``: (N, 3) array [S_tau, S_DoA, S_DoD]; angular spreads must be in (0, 1).
    """
    s = np.atleast_2d(np.asarray(s_obs, float)).copy()
    n = s.shape[0]
    dr = np.broadcast_to(np.asarray(dr_db, float), (n,)).astype(float)
    if np.any(dr <= 0):
        raise ValueError("effective dynamic range must be positive")
    if np.any(s[:, 0] <= 0):
        raise ValueError("delay spread must be positive")
    s[:, 1:] = np.clip(s[:, 1:], 1e-6, 1 - 1e-9)
    if kappa0 is None:
        # infinite-DR approximations: S_tau = k_tau, S_phi ~ sqrt(2) k_phi for narrow Laplacians
        logk = np.column_stack([np.log(s[:, 0]), np.log(s[:, 1] / np.sqrt(2)), np.log(s[:, 2] / np.sqrt(2))])
    else:
        logk = np.log(np.atleast_2d(np.asarray(kappa0, float)))
    h = 1e-30
    f = _forward(logk, dr)
    res = (f - s) / s
    it = 0
    for it in range(1, max_iter + 1):
        err = np.max(np.abs(res), axis=1)
        active = err > tol
        if not np.any(active):
            break
        idx = np.nonzero(active)[0]
        la, da = logk[idx], dr[idx]
        J = np.empty((idx.size, 3, 3))
        for j in range(3):
            lc = la.astype(complex)
            lc[:, j] += 1j * h
            J[:, :, j] = _forward(lc, da).imag / h
        rhs = -(f[idx] - s[idx])
        try:
            step = np.linalg.solve(J, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J.reshape(-1, 3), rhs.reshape(-1), rcond=None)[0].reshape(-1, 3)
        step = np.clip(np.nan_to_num(step), -2.0, 2.0)
        # backtracking: accept the first step size that reduces the residual
        lam = np.ones(idx.size)
        old = err[idx]
        new_l = la.copy()
        new_f = f[idx].copy()
        pending = np.ones(idx.size, bool)
        for _ in range(12):
            cand = np.clip(la + lam[:, None] * step, LOG_KAPPA_MIN, LOG_KAPPA_MAX)
            fc = _forward(cand[pending], da[pending])
            ec = np.max(np.abs((fc - s[idx][pending]) / s[idx][pending]), axis=1)
            ok = (ec < old[pending]) | ~np.isfinite(old[pending])
            pi = np.nonzero(pending)[0]
            acc = pi[ok & np.isfinite(ec)]
            new_l[acc] = cand[pending][ok & np.isfinite(ec)]
            new_f[acc] = fc[ok & np.isfinite(ec)]
            pending[acc] = False
            if not pending.any():
                break
            lam[pending] *= 0.5
        logk[idx] = new_l
        f[idx] = new_f
        res[idx] = (new_f - s[idx]) / s[idx]
    err = np.max(np.abs(res), axis=1)
    return InverseResult(kappa=np.exp(logk), converged=err <= max(tol, 1e-7), iterations=it, residual=err)


# --------------------------------------------------------------------------- discretisation correction
@dataclass
class DiscretizationCorrection:
    """Pre-computed correction matrix Delta(DR, kappa_tau, kappa_R, kappa_T) for one grid."""

    dr_db: NDArray
    log_kb: NDArray          # log of kappa_tau in delay bins
    log_ka: NDArray          # log of kappa_R [rad]
    log_kt: NDArray          # log of kappa_T [rad]
    delta: NDArray           # (n_dr, n_kb, n_ka, n_kt, 3): [Delta S_tau (bins), Delta S_R, Delta S_T]
    delay_res: float
    angle_step_deg: float

    @classmethod
    def build(cls, delay_res: float, angle_step_deg: float, dr_db: ArrayLike | None = None,
              kb: ArrayLike | None = None, ka: ArrayLike | None = None, batch: int = 4000) -> "DiscretizationCorrection":
        dr_db = np.asarray(dr_db if dr_db is not None else
                           [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60], float)
        kb = np.asarray(kb if kb is not None else np.geomspace(0.05, 400.0, 26), float)
        ka = np.asarray(ka if ka is not None else np.geomspace(0.005, 30.0, 24), float)
        grid = AngleGrid(angle_step_deg)
        D, B, A, T = np.meshgrid(dr_db, kb, ka, ka, indexing="ij")
        flat = [x.ravel() for x in (D, B, A, T)]
        out = np.empty((flat[0].size, 3))
        for s in range(0, flat[0].size, batch):
            sl = slice(s, s + batch)
            dsc = discrete_observables(flat[1][sl], flat[2][sl], flat[3][sl], flat[0][sl], grid, delay_res=1.0)
            cnt = continuous_observables(flat[1][sl], flat[2][sl], flat[3][sl], flat[0][sl])
            out[sl] = np.column_stack([dsc[k] - cnt[k] for k in _KEYS])
        delta = out.reshape(D.shape + (3,))
        return cls(dr_db, np.log(kb), np.log(ka), np.log(ka), delta, float(delay_res), float(angle_step_deg))

    # ---- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dr_db=self.dr_db, log_kb=self.log_kb, log_ka=self.log_ka, log_kt=self.log_kt,
                            delta=self.delta, delay_res=self.delay_res, angle_step_deg=self.angle_step_deg)

    @classmethod
    def load(cls, path: str | Path) -> "DiscretizationCorrection":
        z = np.load(path)
        return cls(z["dr_db"], z["log_kb"], z["log_ka"], z["log_kt"], z["delta"], float(z["delay_res"]),
                   float(z["angle_step_deg"]))

    @classmethod
    def load_or_build(cls, path: str | Path, delay_res: float, angle_step_deg: float) -> "DiscretizationCorrection":
        p = Path(path)
        if p.exists():
            c = cls.load(p)
            if abs(c.delay_res - delay_res) < 1e-9 and abs(c.angle_step_deg - angle_step_deg) < 1e-9:
                return c
        c = cls.build(delay_res, angle_step_deg)
        c.save(p)
        return c

    def _interp(self):
        if not hasattr(self, "_rgi"):
            axes = (self.dr_db, self.log_kb, self.log_ka, self.log_kt)
            self._rgi = [RegularGridInterpolator(axes, self.delta[..., i]) for i in range(3)]
        return self._rgi

    def delta_at(self, kappa: NDArray, dr_db: NDArray) -> NDArray:
        """Interpolated Delta (S_tau in the same unit as kappa_tau) at decay constants ``kappa``."""
        kappa = np.atleast_2d(kappa)
        dr = np.broadcast_to(np.asarray(dr_db, float), (kappa.shape[0],))
        pts = np.column_stack([
            np.clip(dr, self.dr_db[0], self.dr_db[-1]),
            np.clip(np.log(kappa[:, 0] / self.delay_res), self.log_kb[0], self.log_kb[-1]),
            np.clip(np.log(kappa[:, 1]), self.log_ka[0], self.log_ka[-1]),
            np.clip(np.log(kappa[:, 2]), self.log_kt[0], self.log_kt[-1]),
        ])
        d = np.column_stack([f(pts) for f in self._interp()])
        d[:, 0] *= self.delay_res
        return d


def solve_decay_constants(s_obs: ArrayLike, dr_db: ArrayLike, correction: DiscretizationCorrection | None = None,
                          n_iter: int = 3, **newton_kw) -> InverseResult:
    """Discretisation correction followed by inverse mapping (step 5 of Sec. VI)."""
    s_obs = np.atleast_2d(np.asarray(s_obs, float))
    res = invert_spreads(s_obs, dr_db, **newton_kw)
    if correction is None:
        return res
    for _ in range(n_iter):
        d = correction.delta_at(res.kappa, dr_db)
        s_c = s_obs - d
        s_c[:, 0] = np.maximum(s_c[:, 0], 1e-3 * correction.delay_res)
        s_c[:, 1:] = np.clip(s_c[:, 1:], 1e-5, 1 - 1e-6)
        res = invert_spreads(s_c, dr_db, kappa0=res.kappa, **newton_kw)
    return res
