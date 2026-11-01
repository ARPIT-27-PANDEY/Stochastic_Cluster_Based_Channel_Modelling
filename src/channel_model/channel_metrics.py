"""Channel-level metrics computed from a complete ADPS (Sec. VIII).

Definitions follow the measurement paper the model is validated against ([21]):
* omni PDP = max over orientation pairs per delay bin ([21] Eq. 3)
* total RMS delay spread = second central moment of the omni PDP ([21] Eq. 7)
* DoA / DoD APS = ADPS summed over delay and the other angle ([21] Eq. 8);
  angular spread = Fleury's definition ([21] Eqs. 9-10)
* path gain = sum over delay of the omni or max-dir PDP ([21] Eqs. 4-5)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .adps import DelayAngleGrid, aps_doa, aps_dod, delay_profile, maxdir_pdp, omni_pdp
from .distributions import fleury_angular_spread, rms_delay_spread


def total_delay_spread(adps: NDArray, grid: DelayAngleGrid, mode: str = "omni") -> float:
    """RMS delay spread [m] of the whole channel; ``mode`` 'omni' (default) or 'sum'."""
    pdp = omni_pdp(adps) if mode == "omni" else delay_profile(adps)
    if pdp.sum() <= 0:
        return float("nan")
    return float(rms_delay_spread(grid.tau_m, pdp))


def total_angular_spreads(adps: NDArray, grid: DelayAngleGrid) -> tuple[float, float]:
    """(DoA, DoD) Fleury angular spreads of the whole channel."""
    phi = grid.phi
    return float(fleury_angular_spread(phi, aps_doa(adps))), float(fleury_angular_spread(phi, aps_dod(adps)))


def path_gain_db(adps: NDArray, mode: str = "omni") -> float:
    """Path gain [dB] = 10 log10 sum_tau PDP_mode(tau), mode in {'omni', 'maxdir'}."""
    pdp = omni_pdp(adps) if mode == "omni" else maxdir_pdp(adps)
    return float(10 * np.log10(pdp.sum()))


def channel_summary(adps: NDArray, grid: DelayAngleGrid) -> dict[str, float]:
    s_doa, s_dod = total_angular_spreads(adps, grid)
    return {
        "delay_spread_m": total_delay_spread(adps, grid, "omni"),
        "delay_spread_sum_m": total_delay_spread(adps, grid, "sum"),
        "doa_spread": s_doa,
        "dod_spread": s_dod,
        "pg_omni_db": path_gain_db(adps, "omni"),
        "pg_maxdir_db": path_gain_db(adps, "maxdir"),
        "n_taps": int(np.count_nonzero(adps)),
    }
