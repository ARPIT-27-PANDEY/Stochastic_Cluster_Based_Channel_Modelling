"""Angular-delay power spectrum (ADPS) utilities (Sec. II, Eq. 1, Eq. 12).

Axis convention used everywhere: ``adps[delay_bin, doa_bin, dod_bin]``
(N_delay x N_R x N_T), i.e. the reshape of APS(phi_T) (x) APS(phi_R) (x) PDP(tau)
in Eq. (12) with the delay index running fastest in the Kronecker vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .distributions import wrap_angle
from .dynamic_range import AngleGrid


@dataclass(frozen=True)
class DelayAngleGrid:
    """Native measurement grid: 1 ns (0.2998 m) delay bins, 10 deg rotor steps."""

    delay_res_m: float = 0.29979
    n_delay: int = 1001
    angle_step_deg: float = 10.0

    @property
    def tau_m(self) -> NDArray:
        return np.arange(self.n_delay) * self.delay_res_m

    @property
    def angles(self) -> AngleGrid:
        return AngleGrid(self.angle_step_deg)

    @property
    def phi(self) -> NDArray:
        return self.angles.phi

    @property
    def n_angle(self) -> int:
        return self.angles.n

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_delay, self.n_angle, self.n_angle)

    @property
    def max_delay_m(self) -> float:
        return self.n_delay * self.delay_res_m

    def angle_index(self, phi: NDArray) -> NDArray:
        """Index of the nearest angular bin (0 deg sits at index n//2)."""
        k = np.round(wrap_angle(phi) / np.deg2rad(self.angle_step_deg)).astype(int)
        return (k + self.n_angle // 2) % self.n_angle


# --------------------------------------------------------------------------- Eq. (1)
def adps_from_frequency_response(H: NDArray, oversampling: int = 1, window: str | None = None) -> NDArray:
    """ADPS of Eq. (1) from frequency responses H[f, phiR, phiT] (complex).

    P(tau, phiR, phiT) = (1/N_f) | sum_k H(phiR, phiT; f_k) e^{j 2 pi f_k tau} |^2.

    With ``oversampling = N_os > 1`` the delay axis is evaluated on an N_os-times finer
    grid (zero padding) and, for every critical delay bin, the maximum over the N_os
    fine points centred on it is kept (Sec. IV-A).  The returned array keeps the native
    Fourier grid (one value per critical bin), so the resolution is not artificially
    increased - only the peak-power under-estimation of off-grid MPCs is corrected.
    The 1/N_f factor makes the critical-grid ADPS satisfy Parseval: sum_tau P = sum_k |H_k|^2.

    ``window`` (None or 'hann') optionally tapers the sweep before the IFFT.  Eq. (1) has no
    window (default); a Hann window lowers the delay sidelobes of strong paths (-13 dB ->
    -31 dB first sidelobe) at the price of a 2x wider main lobe.  Provided only for the
    clustering sensitivity test on synthetic sounder data (docs/METHODOLOGY.md).
    """
    nf = H.shape[0]
    if window == "hann":
        wv = np.hanning(nf)
        H = H * (wv / np.sqrt(np.mean(wv**2)))[(slice(None),) + (None,) * (H.ndim - 1)]
    elif window is not None:
        raise ValueError("window must be None or 'hann'")
    nos = int(oversampling)
    if nos < 1:
        raise ValueError("oversampling must be >= 1")
    h = np.fft.ifft(H, n=nf * nos, axis=0) * (nf * nos)
    p_fine = np.abs(h) ** 2 / nf
    if nos == 1:
        return p_fine
    half = nos // 2
    idx = (np.arange(nf)[:, None] * nos + np.arange(-half, nos - half)[None, :]) % (nf * nos)
    return p_fine[idx].max(axis=1)


# --------------------------------------------------------------------------- Eq. (12)-(13)
def cluster_profiles(kappa_tau_m: float, kappa_r: float, kappa_t: float, grid: DelayAngleGrid,
                     n_delay: int, doa_idx: int = None, dod_idx: int = None) -> tuple[NDArray, NDArray, NDArray]:
    """Return PDP (n_delay), APS_R (N_R), APS_T (N_T) of Eq. (13), peak-normalised to one.

    The angular profiles are Laplacian in the wrapped angle difference to the (grid-snapped)
    centroid bin; the PDP is a one-sided exponential starting at the arrival bin.
    """
    k = np.arange(n_delay)
    pdp = np.exp(-k * grid.delay_res_m / kappa_tau_m)
    phi = grid.phi
    n = grid.n_angle
    c = n // 2
    base_r = np.exp(-np.abs(phi) / kappa_r)
    base_t = np.exp(-np.abs(phi) / kappa_t)
    aps_r = np.roll(base_r, (doa_idx if doa_idx is not None else c) - c)
    aps_t = np.roll(base_t, (dod_idx if dod_idx is not None else c) - c)
    return pdp, aps_r, aps_t


def cluster_adps_kron(pdp: NDArray, aps_r: NDArray, aps_t: NDArray) -> NDArray:
    """Literal Eq. (12): reshape(APS_T (x) APS_R (x) PDP, [N_f, N_R, N_T]).

    numpy's Kronecker vector has its *last* factor (PDP) varying fastest, so a Fortran
    (column-major) reshape to [N_f, N_R, N_T] reproduces the MATLAB-style reshape of the
    paper.  Equivalent to the outer product pdp[:,None,None]*aps_r[None,:,None]*aps_t[None,None,:].
    """
    v = np.kron(aps_t, np.kron(aps_r, pdp))
    return v.reshape((pdp.size, aps_r.size, aps_t.size), order="F")


# --------------------------------------------------------------------------- projections
def delay_profile(adps: NDArray) -> NDArray:
    """Sum over all orientation pairs (angle-integrated PDP)."""
    return adps.sum(axis=(1, 2))


def omni_pdp(adps: NDArray) -> NDArray:
    """Omni-directional PDP of [21, Eq. 3]: per delay bin, the max over orientation pairs."""
    return adps.max(axis=(1, 2))


def maxdir_pdp(adps: NDArray) -> NDArray:
    """PDP of the orientation pair with the largest total power ([21, Eq. 5])."""
    ddaps = adps.sum(axis=0)
    i, j = np.unravel_index(np.argmax(ddaps), ddaps.shape)
    return adps[:, i, j]


def aps_doa(adps: NDArray) -> NDArray:
    return adps.sum(axis=(0, 2))


def aps_dod(adps: NDArray) -> NDArray:
    return adps.sum(axis=(0, 1))
