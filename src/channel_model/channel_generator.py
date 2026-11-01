"""Synthetic channel generation (Sec. VI, Fig. 16, steps 1-7).

For one realization:
 1. inputs: Tx-Rx distance d_LoS, condition ('los'/'nlos'), dynamic range, RNG
 2. number of clusters (LoS cluster handled separately)
 3. per cluster: centroids (arrival, DoA, DoD), observable spreads, power
 4. global cut-off = strongest cluster peak - DR
 5. discretisation correction + inverse mapping -> decay constants kappa
 6. cluster ADPS from kappa (Eq. 12-13), scaled to its power, placed at its centroid
 7. superposition and global thresholding

Coupling of steps 4-6 (implementation decision, see docs/METHODOLOGY.md): the cluster
power drawn from Eq. (8) is an *observable* power (sum of the taps above the cut-off,
Eq. 5), so the peak tap of a cluster depends on its shape, the shape depends on its
effective DR, and DR_eff depends on all peaks via the global cut-off.  We solve this
small fixed point iteratively (warm-started Newton), which converges in a few sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .adps import DelayAngleGrid, cluster_profiles
from .channel_metrics import channel_summary
from .cluster_model import ClusterSet, draw_clusters
from .dynamic_range import discrete_observables
from .inverse_mapping import DiscretizationCorrection, invert_spreads
from .parameters import LOS_CLUSTER, ChannelParameters

_MIN_DR = 1e-3  # dB; clusters whose peak is not above the cut-off are not observable


@dataclass
class ChannelRealization:
    condition: str
    d_los_m: float
    clusters: ClusterSet
    visible: NDArray            # bool per cluster: peak above the cut-off
    kappa: NDArray              # (n, 3) recovered decay constants (nan if not visible)
    dr_eff_db: NDArray          # effective dynamic range per cluster
    peak_db: NDArray            # peak tap power per cluster
    cutoff_db: float
    reobserved: dict[str, NDArray]  # grid-observable spreads of each isolated visible cluster
    fixed_point_iterations: int
    n_dropped_outside_window: int
    adps: NDArray | None = None
    tap_labels: NDArray | None = None   # ground-truth dominant cluster per tap (when kept)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_visible(self) -> int:
        return int(self.visible.sum())


class ChannelGenerator:
    """Python implementation of the synthetic channel generator of Sec. VI."""

    def __init__(self, params: ChannelParameters, grid: DelayAngleGrid | None = None,
                 dynamic_range_db: float = 35.0, correction: DiscretizationCorrection | None = None,
                 use_correction: bool = True, max_fp_iter: int = 30, fp_tol_db: float = 0.01,
                 tail_margin_db: float = 0.0, correction_iterations: int = 3,
                 newton_max_iter: int = 40, newton_tol: float = 1e-8, max_rel_correction: float = 0.5) -> None:
        if dynamic_range_db <= 0:
            raise ValueError("dynamic range must be positive")
        self.p = params
        self.grid = grid or DelayAngleGrid()
        self.dr = float(dynamic_range_db)
        self.correction = correction if use_correction else None
        self.max_fp_iter = max_fp_iter
        self.fp_tol = fp_tol_db
        self.tail_margin_db = tail_margin_db
        self.correction_iterations = int(correction_iterations)
        self.newton_max_iter = int(newton_max_iter)
        self.newton_tol = float(newton_tol)
        self.max_rel_correction = float(max_rel_correction)

    # ------------------------------------------------------------------ step 4-5
    def _invert_corrected(self, s_obs: NDArray, dr_eff: NDArray, kappa0: NDArray | None) -> tuple[NDArray, NDArray]:
        """Discretisation correction + inverse mapping at a *fixed* effective DR (step 5).

        Same scheme as :func:`solve_decay_constants`: Delta is re-evaluated from scratch at the
        current kappa and DR_eff each refinement (no memory of earlier sweeps).  |Delta| is
        bounded to ``max_rel_correction`` of the target spread: the table is a first-order
        correction and, for clusters only a few dB above the cut-off with very narrow angular
        decay, the discrete/continuous mismatch is so large that an unbounded correction
        makes the refinement run away (see docs/METHODOLOGY.md).
        Returns (kappa, Delta used).
        """
        lim = self.max_rel_correction * s_obs
        res = invert_spreads(s_obs, dr_eff, kappa0=kappa0, max_iter=self.newton_max_iter, tol=self.newton_tol)
        d = np.zeros_like(s_obs)
        if self.correction is None:
            return res.kappa, d
        for _ in range(self.correction_iterations):
            d = np.clip(self.correction.delta_at(res.kappa, dr_eff), -lim, lim)
            s_c = s_obs - d
            s_c[:, 1:] = np.minimum(s_c[:, 1:], 0.995)
            res = invert_spreads(s_c, dr_eff, kappa0=res.kappa, max_iter=self.newton_max_iter, tol=self.newton_tol)
        return res.kappa, d

    def _solve_shapes(self, cl: ClusterSet) -> tuple[NDArray, NDArray, NDArray, NDArray, float, int]:
        """Fixed point between the global cut-off and the per-cluster shapes (steps 4-5)."""
        n = cl.n
        s_obs = np.column_stack([cl.s_tau_m, cl.s_doa, cl.s_dod])
        peak = cl.power_db.copy()             # first guess: all power in the peak tap
        cutoff = peak.max() - self.dr
        dr_eff = peak - cutoff
        kappa = np.full((n, 3), np.nan)
        it = 0
        for it in range(1, self.max_fp_iter + 1):
            idx = np.nonzero(dr_eff > _MIN_DR)[0]
            kappa_idx, _ = self._invert_corrected(s_obs[idx], dr_eff[idx], kappa[idx])
            kappa[idx] = kappa_idx
            obs = discrete_observables(kappa_idx[:, 0] / self.grid.delay_res_m, kappa_idx[:, 1], kappa_idx[:, 2],
                                       dr_eff[idx], self.grid.angles, self.grid.delay_res_m)
            shape_sum = np.ones(n)
            shape_sum[idx] = obs["shape_sum"]
            peak = cl.power_db - 10 * np.log10(shape_sum)
            new_cut = peak.max() - self.dr
            change = abs(new_cut - cutoff)
            # damped cut-off update after two plain sweeps: the grid observable is piecewise
            # constant in kappa and two clusters with near-equal peaks can make the plain map 2-cycle
            omega = 1.0 if it <= 2 else max(0.5 / (1.0 + 0.25 * (it - 3)), 0.05)
            cutoff = cutoff + omega * (new_cut - cutoff)
            dr_eff = peak - cutoff
            if change < self.fp_tol:
                break
        vis = dr_eff > _MIN_DR
        # final consistent pass at the final cut-off
        idx = np.nonzero(vis)[0]
        if idx.size:
            k0 = kappa[idx]
            kappa_idx, _ = self._invert_corrected(s_obs[idx], dr_eff[idx], k0)
            kappa[idx] = kappa_idx
            obs = discrete_observables(kappa_idx[:, 0] / self.grid.delay_res_m, kappa_idx[:, 1], kappa_idx[:, 2],
                                       dr_eff[idx], self.grid.angles, self.grid.delay_res_m)
            peak[idx] = cl.power_db[idx] - 10 * np.log10(obs["shape_sum"])
        kappa[~vis] = np.nan
        return kappa, dr_eff, peak, vis, cutoff, it

    # ------------------------------------------------------------------ steps 6-7
    def _build_adps(self, cl: ClusterSet, kappa: NDArray, peak_db: NDArray, dr_eff: NDArray, vis: NDArray,
                    cutoff_db: float, labels_out: list | None = None) -> NDArray:
        """Superpose the cluster ADPSs and apply the global cut-off.

        If ``labels_out`` is a list, a ground-truth label array (index of the cluster that
        contributes most power to each tap, -1 for empty taps) is appended to it.
        """
        g = self.grid
        adps = np.zeros(g.shape)
        best = np.zeros(g.shape) if labels_out is not None else None
        lab = np.full(g.shape, -1, dtype=np.int32) if labels_out is not None else None
        cut_lin = 10 ** (cutoff_db / 10)
        k0 = np.round(cl.arrival_m / g.delay_res_m).astype(int)
        ir = g.angle_index(cl.doa)
        it = g.angle_index(cl.dod)
        for c in np.nonzero(vis)[0]:
            kb = kappa[c, 0] / g.delay_res_m
            n_tail = int(np.floor(kb * (dr_eff[c] + self.tail_margin_db) * np.log(10) / 10)) + 1
            n_tail = max(1, min(n_tail, g.n_delay - k0[c]))
            pdp, ar, at = cluster_profiles(kappa[c, 0], kappa[c, 1], kappa[c, 2], g, n_tail, ir[c], it[c])
            amp = 10 ** (peak_db[c] / 10)
            block = amp * pdp[:, None, None] * ar[None, :, None] * at[None, None, :]
            if self.tail_margin_db <= 0:
                # each cluster contributes only its own observable (above cut-off) taps, which is
                # exactly the truncated shape assumed by the inverse mapping
                block[block < cut_lin] = 0.0
            sl = slice(k0[c], k0[c] + n_tail)
            adps[sl] += block
            if best is not None:
                upd = block > best[sl]
                best[sl][upd] = block[upd]
                lab[sl][upd] = c
        adps[adps < cut_lin] = 0.0
        if lab is not None:
            lab[adps == 0] = -1
            labels_out.append(lab)
        return adps

    # ------------------------------------------------------------------ public API
    def generate(self, d_los_m: float, condition: str, rng: np.random.Generator,
                 keep_adps: bool = False, compute_metrics: bool = True) -> ChannelRealization:
        cl = draw_clusters(condition, d_los_m, self.p, rng)
        inside = cl.arrival_m < self.grid.max_delay_m
        n_drop = int((~inside).sum())
        cl = cl.subset(inside)
        kappa, dr_eff, peak, vis, cutoff, iters = self._solve_shapes(cl)
        reobs = {k: np.full(cl.n, np.nan) for k in ("s_tau", "s_r", "s_t", "n_taps")}
        if vis.any():
            o = discrete_observables(kappa[vis, 0] / self.grid.delay_res_m, kappa[vis, 1], kappa[vis, 2],
                                     dr_eff[vis], self.grid.angles, self.grid.delay_res_m)
            for k in reobs:
                reobs[k][vis] = o[k]
        real = ChannelRealization(condition, float(d_los_m), cl, vis, kappa, dr_eff, peak, float(cutoff), reobs,
                                  iters, n_drop)
        if keep_adps or compute_metrics:
            labs: list | None = [] if keep_adps else None
            adps = self._build_adps(cl, kappa, peak, dr_eff, vis, cutoff, labs)
            if compute_metrics:
                real.metrics = channel_summary(adps, self.grid)
            if keep_adps:
                real.adps = adps
                real.tap_labels = labs[0]
        return real

    def generate_many(self, n: int, condition: str, rng: np.random.Generator, d_range: tuple[float, float] | None = None,
                      d_fixed: float | None = None, keep_adps: bool = False,
                      distances: NDArray | None = None) -> list[ChannelRealization]:
        """Generate ``n`` realizations.

        Distances: ``d_fixed`` if given; else, if ``distances`` (e.g. the measured link
        distances) is given, they are cycled in order; else U(d_range).
        """
        if d_fixed is None:
            d_range = d_range or ((1.0, 100.0) if condition == "los" else (2.0, 100.0))
        out = []
        for i in range(n):
            if d_fixed is not None:
                d = d_fixed
            elif distances is not None:
                d = float(distances[i % len(distances)])
            else:
                d = rng.uniform(*d_range)
            out.append(self.generate(d, condition, rng, keep_adps=keep_adps))
        return out


def ld_setting_visible(real: ChannelRealization, dr_db: float = 15.0) -> NDArray:
    """LoS-excluded, DR-matched (LD) setting of Sec. V-A/B applied to a synthetic realization.

    The LoS cluster is removed and a ``dr_db`` dynamic range is applied relative to the
    strongest remaining cluster peak.  Returns the mask of clusters still observable.
    """
    keep = real.visible & (real.clusters.cls != LOS_CLUSTER)
    if not keep.any():
        return keep
    cut = real.peak_db[keep].max() - dr_db
    return keep & (real.peak_db > cut + _MIN_DR)
