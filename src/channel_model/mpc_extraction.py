"""MPC-tap extraction from the ADPS (Sec. IV-A) and a synthetic sounder for pipeline tests.

The real measured frequency responses are not public (see docs/DATA_AVAILABILITY.md).
:func:`synthetic_frequency_response` builds a *test* data set - discrete specular paths
seen through 13 deg horn patterns on the 10 deg rotor grid over 1001 subcarriers - so
that the Eq. (1) -> thresholding -> clustering pipeline can be exercised end to end.
It is labelled as test data everywhere and is never presented as a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .adps import DelayAngleGrid
from .distributions import wrap_angle


@dataclass
class MPCTaps:
    """Fourier taps of the ADPS inside the dynamic range (the 'MPC taps' of the paper)."""

    delay_idx: NDArray
    doa_idx: NDArray
    dod_idx: NDArray
    tau_m: NDArray
    doa: NDArray
    dod: NDArray
    power: NDArray
    threshold: float

    @property
    def n(self) -> int:
        return int(self.power.size)

    def subset(self, mask: NDArray) -> "MPCTaps":
        return MPCTaps(self.delay_idx[mask], self.doa_idx[mask], self.dod_idx[mask], self.tau_m[mask],
                       self.doa[mask], self.dod[mask], self.power[mask], self.threshold)


def dynamic_range_threshold(adps: NDArray, dr_db: float = 35.0) -> float:
    """Linear power threshold = max(ADPS) - DR (dB)."""
    if dr_db <= 0:
        raise ValueError("dynamic range must be positive")
    return float(adps.max() * 10 ** (-dr_db / 10))


def extract_taps(adps: NDArray, grid: DelayAngleGrid, dr_db: float = 35.0) -> MPCTaps:
    """All grid points whose power is within ``dr_db`` of the ADPS maximum."""
    thr = dynamic_range_threshold(adps, dr_db)
    k, i, j = np.nonzero(adps >= thr)
    phi = grid.phi
    return MPCTaps(k, i, j, k * grid.delay_res_m, phi[i], phi[j], adps[k, i, j], thr)


# --------------------------------------------------------------------------- test data
def horn_amplitude(theta: NDArray, hpbw_deg: float = 13.0) -> NDArray:
    """Gaussian-beam amplitude pattern with the given half-power beamwidth."""
    hp = np.deg2rad(hpbw_deg)
    return np.exp(-2.0 * np.log(2.0) * (wrap_angle(theta) / hp) ** 2)


@dataclass
class TestPaths:
    tau_m: NDArray
    doa: NDArray
    dod: NDArray
    gain: NDArray          # complex amplitude
    cluster: NDArray       # ground-truth cluster id


def make_test_paths(rng: np.random.Generator, n_clusters: int = 4, paths_per_cluster: int = 12,
                    d_los: float = 20.0, delay_offset_m: float = 3.0) -> TestPaths:
    """Clusters of specular paths with exponential power decay - TEST DATA ONLY.

    ``delay_offset_m`` keeps the first path away from delay bin 0: the sinc leakage of
    off-grid paths at negative delays would otherwise wrap round the IFFT window and
    appear as a spurious cluster at ~300 m.
    """
    taus, doas, dods, gains, ids = [], [], [], [], []
    for c in range(n_clusters):
        t0 = delay_offset_m + (0.0 if c == 0 else rng.uniform(5, 80))
        a0 = 0.0 if c == 0 else rng.uniform(-np.pi, np.pi)
        b0 = 0.0 if c == 0 else rng.uniform(-np.pi, np.pi)
        p0 = 0.0 if c == 0 else rng.uniform(-25, -8)
        m = paths_per_cluster
        dt = rng.exponential(0.6, m)
        dt[0] = 0.0
        taus.append(t0 + dt)
        doas.append(wrap_angle(a0 + rng.laplace(0, 0.06, m)))
        dods.append(wrap_angle(b0 + rng.laplace(0, 0.06, m)))
        pw = 10 ** ((p0 - 8.0 * dt / 0.6 / 3 - 3 * np.arange(m) / m) / 10)
        gains.append(np.sqrt(pw) * np.exp(1j * rng.uniform(0, 2 * np.pi, m)))
        ids.append(np.full(m, c))
    return TestPaths(*(np.concatenate(v) for v in (taus, doas, dods, gains, ids)))


def synthetic_frequency_response(paths: TestPaths, grid: DelayAngleGrid, n_freq: int = 1001,
                                 bandwidth_hz: float = 1e9, hpbw_deg: float = 13.0, snr_db: float = 60.0,
                                 rng: np.random.Generator | None = None, c: float = 299792458.0) -> NDArray:
    """H[f, phiR, phiT] of discrete paths observed by rotating horns - TEST DATA ONLY."""
    rng = rng or np.random.default_rng(0)
    f = np.arange(n_freq) * bandwidth_hz / (n_freq - 1)          # baseband offsets of the sweep
    phi = grid.phi
    gr = horn_amplitude(phi[:, None] - paths.doa[None, :], hpbw_deg)   # (NR, P)
    gt = horn_amplitude(phi[:, None] - paths.dod[None, :], hpbw_deg)   # (NT, P)
    ph = np.exp(-2j * np.pi * f[:, None] * paths.tau_m[None, :] / c)  # (F, P)
    H = np.einsum("fp,rp,tp,p->frt", ph, gr, gt, paths.gain, optimize=True)
    p_sig = np.mean(np.abs(H) ** 2)
    noise = np.sqrt(p_sig * 10 ** (-snr_db / 10) / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    return H + noise
