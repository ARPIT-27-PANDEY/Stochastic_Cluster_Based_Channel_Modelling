"""Kernel-power-density (KPD) clustering [54] and the multi-level ML-KPD of Sec. IV-B.

KPD (He et al., IEEE TWC 2017), as summarised in the paper:
 1. K-nearest-neighbour kernel density of every MPC tap, weighted by tap power
 2. relative density = density / max density in the tap's K-neighbourhood
 3. key points = taps whose relative density is 1 (local density maxima)
 4. every other tap follows its nearest higher-density neighbour -> initial clusters
 5. clusters are merged when they are connected through neighbouring taps whose
    relative density exceeds the connectivity threshold

Implementation details that the paper leaves open (documented in docs/METHODOLOGY.md):
 * feature space: delay / std(delay) and, for each angle, the unit phasor scaled by the
   circular standard deviation, so Euclidean distances are wrap-around aware;
 * kernel: Gaussian in that normalised space, weighted by linear tap power / max power;
 * connectivity: two clusters merge if some pair of K-neighbouring taps (one in each)
   both have relative density >= threshold.

ML-KPD (Algorithm 1): K = sqrt(M/2) at level 1; every cluster whose delay, DoA or DoD
projection is multimodal (moving-average smoothing 5/3/3 bins, peaks farther apart than
the window, dip-to-peak ratio < 0.8) is re-clustered with K' = sqrt(M_c/2) and a
threshold searched from 1 downwards in 0.01 steps until the number of sub-clusters does
not exceed the number of detected modes (and keeps at least half of them when possible).
Recursion stops when no cluster is separable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .mpc_extraction import MPCTaps


# --------------------------------------------------------------------------- features
def _circ_std(phi: NDArray, w: NDArray) -> float:
    R = np.abs(np.sum(w * np.exp(1j * phi)) / np.sum(w))
    return float(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 1e-6)))


def kpd_features(taps: MPCTaps) -> NDArray:
    """Normalised 5-D embedding [tau, cos/sin DoA, cos/sin DoD]."""
    w = taps.power / taps.power.max()
    st = float(np.sqrt(np.average((taps.tau_m - np.average(taps.tau_m, weights=w)) ** 2, weights=w)))
    st = max(st, 1e-6)
    sr = _circ_std(taps.doa, w)
    sd = _circ_std(taps.dod, w)
    return np.column_stack([taps.tau_m / st, np.cos(taps.doa) / sr, np.sin(taps.doa) / sr,
                            np.cos(taps.dod) / sd, np.sin(taps.dod) / sd])


# --------------------------------------------------------------------------- KPD
@dataclass
class KPDState:
    """Density structure of one KPD run; labels for any threshold are cheap to derive."""

    knn: NDArray            # (M, K) neighbour indices (self excluded)
    density: NDArray
    rel_density: NDArray
    key: NDArray            # bool
    initial: NDArray        # initial cluster label = index of the key point reached

    def labels(self, threshold: float) -> NDArray:
        """Cluster labels after merging initial clusters connected above ``threshold``."""
        m = self.initial.size
        if self.knn.shape[1] == 0:
            return np.zeros(m, dtype=int)
        if not hasattr(self, "_edges"):
            lab = self.initial
            i_idx = np.repeat(np.arange(m), self.knn.shape[1])
            j_idx = self.knn.ravel()
            cross = lab[i_idx] != lab[j_idx]
            self._edges = (lab[i_idx[cross]], lab[j_idx[cross]],
                           np.minimum(self.rel_density[i_idx[cross]], self.rel_density[j_idx[cross]]))
        a, b, strength = self._edges
        ok = strength >= threshold
        g = coo_matrix((np.ones(int(ok.sum())), (a[ok], b[ok])), shape=(m, m))
        _, comp = connected_components(g, directed=False)
        _, out = np.unique(comp[self.initial], return_inverse=True)
        return out


def kpd_state(taps: MPCTaps, k: int) -> KPDState:
    m = taps.n
    if m == 1:
        return KPDState(np.zeros((1, 0), int), np.ones(1), np.ones(1), np.ones(1, bool), np.zeros(1, int))
    k = int(max(1, min(k, m - 1)))
    X = kpd_features(taps)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    w = taps.power / taps.power.max()
    rho = np.sum(np.exp(w[idx]) * np.exp(-dist**2), axis=1)   # He et al.: exp(normalised power) weighting
    rho = rho + 1e-12 * w  # deterministic tie-break by tap power
    rel = rho / np.maximum(rho, rho[idx].max(axis=1))
    key = rel >= 1.0 - 1e-12
    parent = np.arange(m)
    for i in np.nonzero(~key)[0]:
        higher = idx[i][rho[idx[i]] > rho[i]]
        parent[i] = higher[0] if higher.size else i   # neighbours are sorted by distance
    root = parent.copy()
    for _ in range(m):
        nxt = parent[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    return KPDState(idx, rho, rel, key, root)


def kpd(taps: MPCTaps, k: int | None = None, threshold: float = 0.5) -> NDArray:
    """One KPD clustering pass; ``k`` defaults to sqrt(M/2) (Algorithm 1, line 2)."""
    k = k if k is not None else int(round(np.sqrt(taps.n / 2)))
    return kpd_state(taps, k).labels(threshold)


# --------------------------------------------------------------------------- separability
def count_modes(spectrum: NDArray, window: int, circular: bool = False, dip_ratio: float = 0.8) -> int:
    """Number of distinct modes of a 1-D power spectrum (Sec. IV-B separability test)."""
    s = np.asarray(spectrum, float)
    if s.size == 0 or s.max() <= 0:
        return 0
    if circular:
        s = np.roll(s, -int(np.argmin(s)))   # cut the circle at its minimum
    kern = np.ones(window) / window
    pad = window // 2
    sp = np.concatenate([np.zeros(pad), s, np.zeros(pad)])
    sm = np.convolve(sp, kern, mode="valid")[: s.size]
    peaks = [i for i in range(sm.size)
             if sm[i] > 0 and (i == 0 or sm[i] > sm[i - 1]) and (i == sm.size - 1 or sm[i] >= sm[i + 1])]
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        for a in range(len(peaks) - 1):
            p, q = peaks[a], peaks[a + 1]
            dip = sm[p:q + 1].min()
            low = min(sm[p], sm[q])
            if (q - p) <= window or dip >= dip_ratio * low:
                peaks.pop(a + 1 if sm[p] >= sm[q] else a)
                changed = True
                break
    return len(peaks)


def cluster_modes(taps: MPCTaps, n_angle: int, win_delay: int = 5, win_angle: int = 3,
                  dip_ratio: float = 0.8) -> tuple[int, int, int]:
    """Modes of the delay, DoA and DoD projections of one cluster's taps."""
    k0 = taps.delay_idx.min()
    pd = np.bincount(taps.delay_idx - k0, weights=taps.power)
    pr = np.bincount(taps.doa_idx, weights=taps.power, minlength=n_angle)
    pt = np.bincount(taps.dod_idx, weights=taps.power, minlength=n_angle)
    return (count_modes(pd, win_delay, False, dip_ratio), count_modes(pr, win_angle, True, dip_ratio),
            count_modes(pt, win_angle, True, dip_ratio))


# --------------------------------------------------------------------------- ML-KPD
@dataclass
class MLKPDResult:
    labels: NDArray
    levels: int
    history: list[dict] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return int(self.labels.max() + 1) if self.labels.size else 0


def ml_kpd(taps: MPCTaps, n_angle: int = 36, initial_threshold: float = 0.5, threshold_start: float = 1.0,
           threshold_step: float = 0.01, max_levels: int = 6, min_taps: int = 4, win_delay: int = 5,
           win_angle: int = 3, dip_ratio: float = 0.8) -> MLKPDResult:
    """Multi-level KPD clustering (Algorithm 1)."""
    m = taps.n
    labels = kpd(taps, int(max(1, round(np.sqrt(m / 2)))), initial_threshold)
    hist = [{"level": 1, "n_clusters": int(labels.max() + 1), "K": int(round(np.sqrt(m / 2)))}]
    level = 1
    frozen: set[int] = set()
    while level < max_levels:
        level += 1
        new = labels.copy()
        next_id = labels.max() + 1
        split_any = False
        for c in np.unique(labels):
            if c in frozen:
                continue
            mask = labels == c
            mc = int(mask.sum())
            if mc < min_taps:
                frozen.add(c)
                continue
            sub = taps.subset(mask)
            modes = cluster_modes(sub, n_angle, win_delay, win_angle, dip_ratio)
            n_modes = max(modes)
            if n_modes < 2:
                frozen.add(c)
                continue
            kc = int(max(1, round(np.sqrt(mc / 2))))
            st = kpd_state(sub, kc)
            chosen = None
            t = threshold_start
            while t >= -1e-9:
                lab = st.labels(t)
                nsub = int(lab.max() + 1)
                if nsub <= n_modes:
                    chosen = lab
                    break
                t = round(t - threshold_step, 10)
            if chosen is None or chosen.max() == 0:
                frozen.add(c)
                continue
            split_any = True
            idx = np.nonzero(mask)[0]
            for s in range(1, chosen.max() + 1):
                new[idx[chosen == s]] = next_id
                next_id += 1
        _, labels = np.unique(new, return_inverse=True)
        # frozen ids refer to old labels: recompute by membership
        frozen = set()
        hist.append({"level": level, "n_clusters": int(labels.max() + 1)})
        if not split_any:
            level -= 1
            hist.pop()
            break
    return MLKPDResult(labels, level, hist)
