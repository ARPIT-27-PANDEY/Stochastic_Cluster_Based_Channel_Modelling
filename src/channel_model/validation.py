"""Goodness-of-fit measures and reference-data loaders for model validation (Sec. VIII).

Cramer-von Mises (CvM) distance.  The paper reports "normalized CvM distances" with
reference [62] but gives no formula.  We use the normalised two-sample criterion

    omega^2 = int (F_N(x) - G_M(x))^2 dH_{N+M}(x)
            = 1/(N+M) * sum_{z in pooled sample} (F_N(z) - G_M(z))^2,

i.e. Anderson's (1962) two-sample statistic T divided by its sample-size factor
NM/(N+M).  It is independent of the sample sizes (it estimates the population distance
int (F-G)^2 dH), lies in [0, 1/3], and is the reading under which published values of
~1e-3 ... 3e-2 are meaningful.  ``cvm_T`` returns the un-normalised T for reference
(identical to ``scipy.stats.cramervonmises_2samp(...).statistic``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .parameters import PROJECT_ROOT

DIGITIZED_DIR = PROJECT_ROOT / "data" / "processed" / "digitized"


def _ecdf_at(sample: NDArray, z: NDArray) -> NDArray:
    s = np.sort(sample)
    return np.searchsorted(s, z, side="right") / s.size


def cvm_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Normalised two-sample CvM distance omega^2 (see module docstring)."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("empty sample")
    z = np.concatenate([a, b])
    d = _ecdf_at(a, z) - _ecdf_at(b, z)
    return float(np.mean(d**2))


def cvm_T(a: ArrayLike, b: ArrayLike) -> float:
    """Anderson's two-sample CvM statistic T = NM/(N+M) * omega^2."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return a.size * b.size / (a.size + b.size) * cvm_distance(a, b)


def cvm_pvalue(a: ArrayLike, b: ArrayLike) -> float:
    """Asymptotic/exact p-value of the two-sample CvM test (scipy)."""
    return float(stats.cramervonmises_2samp(np.asarray(a, float), np.asarray(b, float)).pvalue)


def cvm_one_sample(sample: ArrayLike, cdf) -> float:
    """Normalised one-sample distance int (F_N - F)^2 dF  (= W^2 / N)."""
    x = np.sort(np.asarray(sample, float))
    n = x.size
    u = cdf(x)
    w2 = 1.0 / (12 * n) + np.sum((u - (2 * np.arange(1, n + 1) - 1) / (2 * n)) ** 2)
    return float(w2 / n)


def bootstrap_cvm_null(n_a: int, n_b: int, n_boot: int, rng: np.random.Generator) -> NDArray:
    """Distribution of omega^2 between two samples of sizes n_a, n_b from the SAME law.

    Gives the scale of omega^2 that is explained by finite sample size alone.
    """
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = cvm_distance(rng.random(n_a), rng.random(n_b))
    return out


# --------------------------------------------------------------------------- digitized data
def load_manifest() -> dict:
    p = DIGITIZED_DIR / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("digitized curves missing - run scripts/digitize_published_figures.py --pdf <paper.pdf>")
    return json.loads(p.read_text())


def digitized_available() -> bool:
    return (DIGITIZED_DIR / "manifest.json").exists()


def load_curve(figure: str, role: str) -> tuple[NDArray, NDArray]:
    a = np.loadtxt(DIGITIZED_DIR / f"{figure}__{role}.csv", delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def load_samples(figure: str, role: str) -> NDArray:
    return np.loadtxt(DIGITIZED_DIR / f"{figure}__{role}__samples.csv", delimiter=",", skiprows=1)


def curve_deviation(x: NDArray, y: NDArray, cdf) -> float:
    """Max |y - cdf(x)| between a digitized curve and a model CDF (calibration check)."""
    return float(np.max(np.abs(np.asarray(y) - cdf(np.asarray(x)))))
