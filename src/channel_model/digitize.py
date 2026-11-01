"""Exact extraction of curves from the *vector* figures of the paper PDF.

Several figures of the paper (Figs. 2, 6, 10, 11, 17) are embedded as vector graphics,
so the plotted polylines can be read back exactly (no image tracing).  This is how the
published *measured* empirical CDFs are recovered for comparison - they are labelled
"digitized from the published figure" everywhere and are never presented as raw data.

The PDF itself is not part of the repository (IEEE copyright); users point the script
``scripts/digitize_published_figures.py`` at their own copy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Panel:
    """A plot panel: page number (1-based), clip rectangle in PDF points, axis ranges."""

    name: str
    page: int
    clip: tuple[float, float, float, float]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_log: bool = False


def _frame(drawings, clip) -> tuple[float, float, float, float]:
    """Axes frame = bounding box of the dark (0.15 gray) axis lines inside the clip."""
    xs, ys = [], []
    for d in drawings:
        col = d.get("color")
        r = d["rect"]
        if col is None or not (r.x0 >= clip[0] and r.x1 <= clip[2] and r.y0 >= clip[1] and r.y1 <= clip[3]):
            continue
        if all(abs(c - 0.15) < 0.02 for c in col):
            xs += [r.x0, r.x1]
            ys += [r.y0, r.y1]
    if not xs:
        raise RuntimeError("axes frame not found")
    return min(xs), min(ys), max(xs), max(ys)


def _points(d) -> NDArray:
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
    a = np.array(pts, float)
    if a.size == 0:
        return a.reshape(0, 2)
    keep = np.ones(len(a), bool)
    keep[1:] = np.any(np.abs(np.diff(a, axis=0)) > 1e-6, axis=1)
    return a[keep]


def extract_curves(pdf_path: str, panel: Panel, min_items: int = 4) -> list[dict]:
    """Return every coloured polyline in ``panel`` mapped to data coordinates."""
    import pymupdf  # optional dependency, only needed for digitising

    doc = pymupdf.open(pdf_path)
    page = doc[panel.page - 1]
    dr = page.get_drawings()
    x0, y0, x1, y1 = _frame(dr, panel.clip)
    out = []
    for d in dr:
        col = d.get("color")
        r = d["rect"]
        if col is None or len(d["items"]) < min_items:
            continue
        if not (r.x0 >= panel.clip[0] and r.x1 <= panel.clip[2] and r.y0 >= panel.clip[1] and r.y1 <= panel.clip[3]):
            continue
        if max(col) - min(col) < 0.05:   # grey / black = grid, axes, text
            continue
        p = _points(d)
        fx = (p[:, 0] - x0) / (x1 - x0)
        if panel.x_log:
            lx = np.log10(panel.x_range[0]) + fx * (np.log10(panel.x_range[1]) - np.log10(panel.x_range[0]))
            xd = 10**lx
        else:
            xd = panel.x_range[0] + fx * (panel.x_range[1] - panel.x_range[0])
        yd = panel.y_range[0] + (y1 - p[:, 1]) / (y1 - y0) * (panel.y_range[1] - panel.y_range[0])
        out.append({"color": tuple(round(c, 3) for c in col), "dashes": d.get("dashes"), "width": d.get("width"),
                    "x": xd, "y": yd, "n_items": len(d["items"])})
    return out


def infer_sample_size(y: NDArray) -> int:
    """Sample size of an empirical step CDF = 1 / (smallest jump height)."""
    dy = np.diff(y)
    return int(round(1.0 / dy[dy > 1e-4].min()))


def step_cdf_to_samples(x: NDArray, y: NDArray, n_samples: int | None = None,
                        x_max: float | None = None) -> tuple[NDArray, int]:
    """Recover the sample values of an empirical step CDF.

    Every upward jump of the polyline at abscissa x_j of height dy_j contributes
    round(dy_j * n) samples equal to x_j.  If the curve ends below one (samples beyond
    the plotted axis range), the missing samples are set to ``x_max`` and their number
    is returned so the caller can report it.
    """
    n = n_samples or infer_sample_size(y)
    samples: list[float] = []
    for i in range(1, len(x)):
        dy = y[i] - y[i - 1]
        if dy > 0.5 / n:
            samples += [0.5 * (x[i] + x[i - 1])] * int(round(dy * n))
    n_clip = n - len(samples)
    if n_clip < 0 or (n_clip > 0 and x_max is None):
        raise ValueError(f"recovered {len(samples)} samples, expected {n}")
    samples += [float(x_max)] * n_clip
    return np.sort(np.array(samples)), n_clip
