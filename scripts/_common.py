"""Shared helpers for the command-line scripts: configuration, generator construction,
deterministic (parallel) synthetic data sets with caching, logging and figure style."""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from channel_model import (ChannelGenerator, DelayAngleGrid, DiscretizationCorrection,  # noqa: E402
                           load_channel_parameters, load_simulation_config)
from channel_model.channel_generator import ld_setting_visible  # noqa: E402
from channel_model.parameters import get_profile  # noqa: E402
from channel_model.power_model import fspl_db  # noqa: E402

RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"
EXP_DIR = RESULTS / "experiments"
LOG_DIR = RESULTS / "logs"
SYN_DIR = ROOT / "data" / "synthetic"
for _d in (FIG_DIR, TAB_DIR, EXP_DIR, LOG_DIR, SYN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COND_ID = {"los": 0, "nlos": 1}
TAG_ID = {"val": 0, "pl": 1, "mle": 2, "perf": 3, "misc": 9}


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(LOG_DIR / f"{name}.log", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(sh)
        log.addHandler(fh)
    return log


def environment_info() -> dict:
    import matplotlib
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__, "cpu_count": os.cpu_count()}


def load_all(profile: str) -> tuple[dict, dict, dict]:
    sim = load_simulation_config()
    return sim, get_profile(profile, sim), sim["generator"]


def correction_table(sim: dict) -> DiscretizationCorrection:
    g = sim["grid"]
    return DiscretizationCorrection.load_or_build(ROOT / sim["generator"]["correction_table"],
                                                  g["delay_resolution_m"], g["angle_step_deg"])


def make_generator(sim: dict, dr_db: float | None = None, use_correction: bool = True,
                   params=None) -> ChannelGenerator:
    g, gc = sim["grid"], sim["generator"]
    grid = DelayAngleGrid(g["delay_resolution_m"], g["n_delay_bins"], g["angle_step_deg"])
    corr = correction_table(sim) if use_correction else None
    return ChannelGenerator(params or load_channel_parameters(), grid, dr_db if dr_db is not None else gc["dynamic_range_db"],
                            corr, use_correction, gc["max_fixed_point_iterations"], gc["fixed_point_tol_db"],
                            g["tail_margin_db"], gc["correction_iterations"], gc["newton_max_iter"], gc["newton_tol"])


# --------------------------------------------------------------------------- data sets
_WORKER: dict = {}


def _init_worker(dr_db: float, use_correction: bool, variant: dict | None = None) -> None:
    sim = load_simulation_config()
    params = load_channel_parameters(**variant) if variant else None
    _WORKER["gen"] = make_generator(sim, dr_db, use_correction, params)


def realization_seed(seed: int, cond: str, tag: str, i: int) -> np.random.Generator:
    """Independent stream per realization: results do not depend on worker count/order."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(COND_ID[cond], TAG_ID[tag], i)))


def _one(args) -> tuple[dict, list[dict]]:
    i, seed, cond, tag, d = args
    gen: ChannelGenerator = _WORKER["gen"]
    rng = realization_seed(seed, cond, tag, i)
    t0 = time.perf_counter()
    r = gen.generate(d, cond, rng, keep_adps=False, compute_metrics=True)
    dt = time.perf_counter() - t0
    ld = ld_setting_visible(r, 15.0)
    cl = r.clusters
    row = {"rid": i, "condition": cond, "d_m": d, "n_clusters": cl.n, "n_visible": r.n_visible,
           "n_visible_ld15": int(ld.sum()), "n_dropped_window": r.n_dropped_outside_window,
           "fp_iter": r.fixed_point_iterations, "cutoff_db": r.cutoff_db, "time_s": dt, **r.metrics}
    thr_ncp = r.cutoff_db + float(fspl_db(d, gen.p.fc_hz))
    recs = []
    for c in range(cl.n):
        recs.append({"rid": i, "cls": int(cl.cls[c]), "arrival_m": cl.arrival_m[c], "doa": cl.doa[c], "dod": cl.dod[c],
                     "s_tau_m": cl.s_tau_m[c], "s_doa": cl.s_doa[c], "s_dod": cl.s_dod[c], "ratio_db": cl.ratio_db[c],
                     "ncp_db": cl.ncp_db[c], "power_db": cl.power_db[c], "peak_db": r.peak_db[c],
                     "dr_eff_db": r.dr_eff_db[c], "visible": bool(r.visible[c]), "visible_ld15": bool(ld[c]),
                     "thr_ncp_db": thr_ncp, "kappa_tau_m": r.kappa[c, 0], "kappa_r": r.kappa[c, 1],
                     "kappa_t": r.kappa[c, 2], "re_s_tau_m": r.reobserved["s_tau"][c],
                     "re_s_doa": r.reobserved["s_r"][c], "re_s_dod": r.reobserved["s_t"][c],
                     "re_n_taps": r.reobserved["n_taps"][c]})
    return row, recs


def simulate(cond: str, n: int, seed: int, tag: str, distances: np.ndarray, dr_db: float = 35.0,
             use_correction: bool = True, workers: int | None = None, cache: str | None = None,
             log: logging.Logger | None = None, variant: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ``n`` realizations at the given distances (cycled) and return
    (realization table, cluster table).  Cached under data/synthetic/<cache>_*.csv.gz."""
    if cache:
        f1, f2 = SYN_DIR / f"{cache}_real.csv.gz", SYN_DIR / f"{cache}_clusters.csv.gz"
        if f1.exists() and f2.exists():
            r, c = pd.read_csv(f1), pd.read_csv(f2)
            if len(r) == n:
                if log:
                    log.info("loaded cached data set %s (%d realizations)", cache, n)
                return r, c
    sim = load_simulation_config()
    correction_table(sim)   # build once in the parent before forking workers
    jobs = [(i, seed, cond, tag, float(distances[i % len(distances)])) for i in range(n)]
    workers = workers or max(1, (os.cpu_count() or 2) - 0)
    t0 = time.perf_counter()
    rows, recs = [], []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(dr_db, use_correction, variant)) as ex:
        for k, (row, rr) in enumerate(ex.map(_one, jobs, chunksize=max(1, n // (8 * workers)))):
            rows.append(row)
            recs.extend(rr)
            if log and (k + 1) % max(1, n // 10) == 0:
                log.info("%s %s: %d/%d realizations (%.0f s)", cache or tag, cond, k + 1, n, time.perf_counter() - t0)
    r, c = pd.DataFrame(rows), pd.DataFrame(recs)
    if cache:
        r.to_csv(f1, index=False)
        c.to_csv(f2, index=False)
    return r, c


def validation_distances(params, cond: str) -> np.ndarray:
    m = params.raw["measurement"] if hasattr(params, "raw") else None
    key = "measured_distances_los_m" if cond == "los" else "measured_distances_nlos_m"
    if m is None:
        import yaml
        m = yaml.safe_load((ROOT / "config" / "channel_parameters.yaml").read_text())["measurement"]
    return np.asarray(m[key], float)


def uniform_distances(n: int, seed: int, cond: str, lo: float = 1.0, hi: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(COND_ID[cond], TAG_ID["pl"], 10**6)))
    return rng.uniform(lo, hi, n)


# --------------------------------------------------------------------------- figure style
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 7.5, "legend.frameon": False, "savefig.bbox": "tight"})
    return plt


COLORS = {"los": "#1f77b4", "nlos": "#d62728", "loscluster": "#2ca02c", "paper": "#7f7f7f", "ours": "#ff7f0e"}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
