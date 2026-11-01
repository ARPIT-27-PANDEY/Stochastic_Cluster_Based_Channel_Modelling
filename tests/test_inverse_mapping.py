"""Inverse mapping observable spreads -> decay constants and the discretisation correction."""
from pathlib import Path

import numpy as np
import pytest

from channel_model.dynamic_range import AngleGrid, continuous_observables, discrete_observables
from channel_model.inverse_mapping import DiscretizationCorrection, invert_spreads, solve_decay_constants

TABLE = Path(__file__).resolve().parents[1] / "data" / "processed" / "discretization_correction.npz"


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    n = 200
    kappa = np.column_stack([rng.uniform(0.2, 30, n), np.exp(rng.uniform(np.log(0.02), np.log(2), n)),
                             np.exp(rng.uniform(np.log(0.02), np.log(2), n))])
    dr = rng.uniform(3, 40, n)
    o = continuous_observables(kappa[:, 0], kappa[:, 1], kappa[:, 2], dr)
    s = np.column_stack([o["s_tau"], o["s_r"], o["s_t"]])
    res = invert_spreads(s, dr)
    assert np.all(res.converged)
    assert np.allclose(res.kappa, kappa, rtol=1e-5)


def test_figure15_point():
    # kappa_DoD = 0.2 pi, kappa_DoA = 0.3 pi, DR = 25 dB
    o = continuous_observables(10.0, 0.3 * np.pi, 0.2 * np.pi, 25.0)
    res = invert_spreads(np.column_stack([o["s_tau"], o["s_r"], o["s_t"]]), [25.0])
    assert np.allclose(res.kappa[0], [10.0, 0.3 * np.pi, 0.2 * np.pi], rtol=1e-6)


def test_invalid_angular_spread_flagged_or_rejected():
    # an angular spread of ~1 cannot be produced under a 3 dB DR -> not converged / saturated
    res = invert_spreads(np.array([[1.0, 0.99, 0.99]]), [3.0])
    o = continuous_observables(*res.kappa[0], 3.0)
    assert (not res.converged[0]) or abs(o["s_r"][0] - 0.99) < 1e-3


@pytest.mark.skipif(not TABLE.exists(), reason="correction table not built (scripts/generate_channels.py builds it)")
def test_correction_reduces_discrete_error():
    corr = DiscretizationCorrection.load(TABLE)
    g = AngleGrid(10.0)
    res_m = 0.29979
    rng = np.random.default_rng(1)
    target = np.column_stack([rng.uniform(1, 8, 40), rng.uniform(0.1, 0.4, 40), rng.uniform(0.1, 0.4, 40)])
    dr = rng.uniform(10, 35, 40)
    err = {}
    for name, c in (("plain", None), ("corrected", corr)):
        r = solve_decay_constants(target, dr, c)
        d = discrete_observables(r.kappa[:, 0] / res_m, r.kappa[:, 1], r.kappa[:, 2], dr, g, res_m)
        got = np.column_stack([d["s_tau"], d["s_r"], d["s_t"]])
        err[name] = np.median(np.abs(got / target - 1), axis=0)
    assert np.all(err["corrected"] <= err["plain"] + 1e-3)
    assert np.all(err["corrected"] < 0.05)
