"""Random drawing of cluster parameters from the Table I model (Sec. V, steps 2-3 of Sec. VI)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import distributions as dist
from .parameters import CLASS_KEYS, LOS, LOS_CLUSTER, NLOS, AngleModel, ChannelParameters
from .power_model import distance_ratio_db, ncp_to_cluster_power_db, sample_ncp


@dataclass
class ClusterSet:
    """Cluster parameters of one channel realization (one entry per drawn cluster)."""

    cls: NDArray            # cluster class: 0 LoS cluster, 1 other LoS, 2 NLoS
    arrival_m: NDArray      # excess run length of the earliest tap relative to d_LoS
    doa: NDArray            # centroid DoA [rad] in [-pi, pi)
    dod: NDArray            # centroid DoD [rad]
    s_tau_m: NDArray        # drawn observable intra-cluster delay spread [m]
    s_doa: NDArray          # drawn observable intra-cluster DoA angular spread
    s_dod: NDArray
    ratio_db: NDArray       # 10 log10(d/d_LoS), d = d_LoS + arrival
    ncp_db: NDArray         # normalised cluster power
    power_db: NDArray       # [P_c]dB = NCP - FSPL(d_LoS)

    @property
    def n(self) -> int:
        return int(self.cls.size)

    def subset(self, mask: NDArray) -> "ClusterSet":
        return ClusterSet(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})


def draw_cluster_count(condition: str, p: ChannelParameters, rng: np.random.Generator) -> tuple[int, int]:
    """Return (number of dominant LoS clusters, number of other clusters)."""
    if condition == "los":
        n_add = int(dist.round_cluster_count(dist.gamma_sample(p.n_los_additional.shape, p.n_los_additional.scale, 1, rng))[0])
        return 1, max(n_add, 0)
    if condition == "nlos":
        n_m1 = int(dist.round_cluster_count(dist.gamma_sample(p.n_nlos_minus_one.shape, p.n_nlos_minus_one.scale, 1, rng))[0])
        return 0, max(n_m1, 0) + 1
    raise ValueError("condition must be 'los' or 'nlos'")


def draw_centroids(model: AngleModel, n: int, rng: np.random.Generator, mode: str) -> NDArray:
    if model.type == "fixed":
        return np.full(n, model.value)
    if model.type == "uniform":
        return dist.uniform_circular_sample(n, rng)
    return dist.bimodal_laplace_sample(model.weights, model.centers, model.decays, n, rng, mode=mode)


def draw_clusters(condition: str, d_los_m: float, p: ChannelParameters, rng: np.random.Generator) -> ClusterSet:
    """Steps 2-3 of the generation procedure: number of clusters and all per-cluster draws."""
    if d_los_m <= 0:
        raise ValueError("Tx-Rx distance must be positive")
    n_dom, n_oth = draw_cluster_count(condition, p, rng)
    other = LOS if condition == "los" else NLOS
    cls = np.array([LOS_CLUSTER] * n_dom + [other] * n_oth, dtype=int)
    n = cls.size
    arrival = np.zeros(n)
    doa = np.zeros(n)
    dod = np.zeros(n)
    s_tau = np.zeros(n)
    s_r = np.zeros(n)
    s_t = np.zeros(n)
    for c in (LOS_CLUSTER, LOS, NLOS):
        m = cls == c
        k = int(m.sum())
        if k == 0:
            continue
        key = CLASS_KEYS[c]
        if c != LOS_CLUSTER:
            g = p.arrival[key]
            arrival[m] = dist.gamma_sample(g.shape, g.scale, k, rng)
        doa[m] = draw_centroids(p.doa_centroid[key], k, rng, p.bimodal_laplace_mode)
        dod[m] = draw_centroids(p.dod_centroid[key], k, rng, p.bimodal_laplace_mode)
        s_tau[m] = dist.lognormal_sample(p.delay_spread[key].mu, p.delay_spread[key].sigma, k, rng)
        s_r[m] = dist.lognormal_sample(p.doa_spread[key].mu, p.doa_spread[key].sigma, k, rng)
        s_t[m] = dist.lognormal_sample(p.dod_spread[key].mu, p.dod_spread[key].sigma, k, rng)
    s_r = np.minimum(s_r, 0.999)   # Fleury spread is bounded by one
    s_t = np.minimum(s_t, 0.999)
    ratio = distance_ratio_db(d_los_m + arrival, d_los_m)
    ncp = np.zeros(n)
    for c in (LOS_CLUSTER, LOS, NLOS):
        m = cls == c
        if m.any():
            ncp[m] = sample_ncp(ratio[m], p.power[CLASS_KEYS[c]], rng, p.min_sigma_db)
    power = ncp_to_cluster_power_db(ncp, d_los_m, p.fc_hz)
    return ClusterSet(cls, arrival, doa, dod, s_tau, s_r, s_t, ratio, ncp, power)
