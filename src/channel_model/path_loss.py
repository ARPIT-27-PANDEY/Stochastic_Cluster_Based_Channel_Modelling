"""Path-loss model fitting with 95 % confidence intervals (Sec. VIII, Table II).

* alpha-beta model:  PL(d) = alpha + 10 beta log10(d / 1 m) + X_sigma          ([21] Eq. 6)
* close-in model:    PL(d) = FSPL(1 m) + 10 n log10(d / 1 m) + X_sigma        ([63] Eq. 2)

Confidence intervals: Student-t for the regression coefficients and chi-square for the
shadowing standard deviation (dof = N-2 for alpha-beta, N-1 for CIM).  These choices
reproduce the widths of the intervals printed in Table II (e.g. sigma = 1.98 with
N = 50 gives [1.65, 2.47], Table II: [1.66, 2.47]).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .power_model import fspl_db


@dataclass
class AlphaBetaFit:
    alpha: float
    alpha_ci: tuple[float, float]
    beta: float
    beta_ci: tuple[float, float]
    sigma: float
    sigma_ci: tuple[float, float]
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CloseInFit:
    n_exp: float
    n_ci: tuple[float, float]
    sigma: float
    sigma_ci: tuple[float, float]
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def _sigma_ci(sse: float, dof: int, level: float) -> tuple[float, float]:
    a = 1 - level
    return (float(np.sqrt(sse / stats.chi2.ppf(1 - a / 2, dof))), float(np.sqrt(sse / stats.chi2.ppf(a / 2, dof))))


def fit_alpha_beta(d_m: ArrayLike, pl_db: ArrayLike, level: float = 0.95) -> AlphaBetaFit:
    d = np.asarray(d_m, float)
    y = np.asarray(pl_db, float)
    if d.size < 3:
        raise ValueError("need at least three points")
    x = 10 * np.log10(d)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = d.size - 2
    sse = float(res @ res)
    s2 = sse / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    t = stats.t.ppf(0.5 + level / 2, dof)
    se = np.sqrt(np.diag(cov))
    return AlphaBetaFit(float(coef[0]), (float(coef[0] - t * se[0]), float(coef[0] + t * se[0])),
                        float(coef[1]), (float(coef[1] - t * se[1]), float(coef[1] + t * se[1])),
                        float(np.sqrt(s2)), _sigma_ci(sse, dof, level), int(d.size))


def fit_close_in(d_m: ArrayLike, pl_db: ArrayLike, fc_hz: float, level: float = 0.95) -> CloseInFit:
    d = np.asarray(d_m, float)
    y = np.asarray(pl_db, float) - fspl_db(1.0, fc_hz)
    x = 10 * np.log10(d)
    n_exp = float(np.sum(x * y) / np.sum(x * x))
    res = y - n_exp * x
    dof = d.size - 1
    sse = float(res @ res)
    s = np.sqrt(sse / dof)
    t = stats.t.ppf(0.5 + level / 2, dof)
    se = s / np.sqrt(np.sum(x * x))
    return CloseInFit(n_exp, (n_exp - t * se, n_exp + t * se), float(s), _sigma_ci(sse, dof, level), int(d.size))


def ci_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])
