"""Helpers shared by the GNU Radio flowgraphs in ``gnuradio/``.

The flowgraphs implement the paper's measurement and generation chain in GNU Radio:

* ``subthz_sounder_batch.grc`` (no GUI) - a 145-146 GHz VNA channel sounder in GNU Radio.
  A generator block draws a double-directional channel from the cluster model (Sec. VI),
  and for each of the 36 x 36 Tx/Rx horn orientations emits the 1001-tone frequency
  response H(f).  Standard GNU Radio blocks add the sounder noise, take the inverse FFT
  and |.|^2 (Eq. 1); a sink block assembles the ADPS, applies the 35 dB MPC threshold
  (Sec. IV-A) and computes delay / angular spreads and path gains (Sec. VIII).
* ``subthz_channel_emulator.grc`` (Qt GUI) - interactive emulator: PDP / APS displays and a
  wideband test signal passed through the generated channel impulse response with the
  standard GNU Radio FIR filter.

This module holds the numpy parts so the embedded-Python blocks stay small.  It does not
import GNU Radio itself.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

from .adps import DelayAngleGrid, omni_pdp
from .channel_generator import ChannelGenerator
from .channel_metrics import channel_summary
from .inverse_mapping import DiscretizationCorrection
from .parameters import PROJECT_ROOT, load_channel_parameters, load_simulation_config

COND_ID = {"los": 0, "nlos": 1}
NF = 1001                      # tones per sweep (145-146 GHz, 1 MHz spacing)
DR_DB = 35.0                   # MPC threshold below the ADPS maximum


def resolve_params_file(params_file: str | None) -> Path | None:
    if not params_file:
        return None
    p = Path(params_file)
    return p if p.is_absolute() else PROJECT_ROOT / p


@functools.lru_cache(maxsize=4)
def get_generator(params_file: str = "") -> ChannelGenerator:
    sim = load_simulation_config()
    g, gc = sim["grid"], sim["generator"]
    grid = DelayAngleGrid(g["delay_resolution_m"], g["n_delay_bins"], g["angle_step_deg"])
    corr = DiscretizationCorrection.load(PROJECT_ROOT / gc["correction_table"])
    params = load_channel_parameters(resolve_params_file(params_file))
    return ChannelGenerator(params, grid, gc["dynamic_range_db"], corr, True, gc["max_fixed_point_iterations"],
                            gc["fixed_point_tol_db"], g["tail_margin_db"], gc["correction_iterations"],
                            gc["newton_max_iter"], gc["newton_tol"])


def measured_distances(cond: str) -> np.ndarray:
    import yaml
    m = yaml.safe_load((PROJECT_ROOT / "config" / "channel_parameters.yaml").read_text())["measurement"]
    return np.asarray(m["measured_distances_los_m" if cond == "los" else "measured_distances_nlos_m"], float)


def link_distance(cond: str, mode: str, seed: int, i: int) -> float:
    """'measured': cycle through the 21/17 measured link distances of [21];
    'uniform': d ~ U(1, 100) m (Table II experiment)."""
    if mode == "measured":
        d = measured_distances(cond)
        return float(d[i % d.size])
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(COND_ID[cond], 51, i)))
    return float(rng.uniform(1.0, 100.0))


def realization_rng(seed: int, cond: str, i: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(COND_ID[cond], 50, i)))


@functools.lru_cache(maxsize=8)
def realization(cond: str, d_m: float, seed: int, index: int, params_file: str = ""):
    gen = get_generator(params_file)
    return gen.generate(d_m, cond, realization_rng(seed, cond, index), keep_adps=True, compute_metrics=True)


# --------------------------------------------------------------------------- sounder side
def sounder_sweeps(adps: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Frequency responses seen by a VNA sweep for every horn orientation pair.

    Each ADPS tap becomes a complex path amplitude sqrt(P) e^{j theta} (random phase), and
    H(f_k) = (1/sqrt(N_f)) sum_n h_n e^{-j 2 pi k n / N_f}.  With this normalisation the
    GNU Radio chain  IFFT -> |.|^2 -> x(1/N_f)  returns exactly P (Eq. 1, Parseval).
    The sweeps are normalised so that the strongest tap is 0 dB; the removed gain is
    returned (dB) and re-applied to the path gains by the sink.

    Returns H with shape (N_R * N_T, N_f), ordered DoA-major (index = i_doa * N_T + i_dod).
    """
    nd, nr, nt = adps.shape
    peak = float(adps.max())
    gain_db = 10 * np.log10(peak) if peak > 0 else 0.0
    p = (adps / peak) if peak > 0 else adps
    h = np.sqrt(p) * np.exp(1j * rng.uniform(0, 2 * np.pi, adps.shape))
    h = np.transpose(h, (1, 2, 0)).reshape(nr * nt, nd)
    H = np.fft.fft(h, n=NF, axis=1) / np.sqrt(NF)
    return H.astype(np.complex64), gain_db


def metrics_from_sounding(pdp_rows: np.ndarray, gain_db: float, grid: DelayAngleGrid | None = None,
                          dr_db: float = DR_DB) -> dict:
    """ADPS assembly + 35 dB MPC threshold + channel metrics (sink side).

    ``pdp_rows`` has shape (N_R * N_T, N_f): the |IFFT|^2 / N_f output of the GNU Radio chain.
    """
    grid = grid or DelayAngleGrid()
    nr = nt = grid.n_angle
    adps = pdp_rows.reshape(nr, nt, -1)[:, :, :grid.n_delay].transpose(2, 0, 1).astype(float)
    adps = np.where(adps >= adps.max() * 10 ** (-dr_db / 10), adps, 0.0)
    m = channel_summary(adps, grid)
    m["pg_omni_db"] += gain_db
    m["pg_maxdir_db"] += gain_db
    return m


def truth_tag(real) -> str:
    keep = ("delay_spread_m", "doa_spread", "dod_spread", "pg_omni_db", "pg_maxdir_db", "n_taps")
    return json.dumps({**{k: float(real.metrics[k]) for k in keep}, "n_visible": int(real.n_visible),
                       "n_clusters": int(real.clusters.n)})


# --------------------------------------------------------------------------- emulator side
def emulator_views(real) -> dict:
    """Display vectors and FIR taps for the Qt emulator."""
    adps = real.adps
    floor = adps[adps > 0].min() if np.any(adps > 0) else 1e-30
    db = lambda x: 10 * np.log10(np.maximum(x, floor * 1e-2))
    pdp = omni_pdp(adps)
    aps_r = adps.sum(axis=(0, 2))
    aps_t = adps.sum(axis=(0, 1))
    rng = np.random.default_rng(12345)
    taps = np.sqrt(pdp / pdp.max()) * np.exp(1j * rng.uniform(0, 2 * np.pi, pdp.size))
    last = int(np.nonzero(pdp)[0].max()) + 1 if np.any(pdp > 0) else 1
    return {"pdp_db": db(pdp).astype(np.float32), "aps_r_db": db(aps_r).astype(np.float32),
            "aps_t_db": db(aps_t).astype(np.float32), "taps": taps[:last].astype(np.complex64)}


def emulator_summary(real) -> str:
    m = real.metrics
    return (f"{real.condition.upper()} d = {real.d_los_m:.1f} m | "
            f"clusters drawn {real.clusters.n}, observable {real.n_visible} | "
            f"DS {m['delay_spread_m']:.2f} m ({m['delay_spread_m'] / 0.29979:.1f} ns) | "
            f"ASA {m['doa_spread']:.3f}  ASD {m['dod_spread']:.3f} | "
            f"PG omni {m['pg_omni_db']:.1f} dB, max-dir {m['pg_maxdir_db']:.1f} dB")


def repo_root() -> str:
    return str(PROJECT_ROOT)


if os.environ.get("SUBTHZ_DEBUG"):
    print("gr_support loaded from", __file__)
