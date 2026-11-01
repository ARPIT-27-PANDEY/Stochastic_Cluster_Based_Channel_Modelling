"""Generate the synthetic channel data sets used by validation, figures and experiments.

    python scripts/generate_channels.py --profile smoke|standard|full [--workers 4]
    python scripts/generate_channels.py --example --condition los --distance 20 --seed 1

Data sets (all SIMULATED with the published Table I parameters; none is measured data):
  <profile>_val_<cond> : n_realizations at the 21 LoS / 17 NLoS measured link distances of
                         [21] Table II, cycled  -> Fig. 17 / CvM validation, Figs 2-13 counterparts
  <profile>_pl_<cond>  : n_realizations at distances ~ U(1, 100) m -> path-loss pool (Table II)
The discretisation-correction table is built on first use (~2 min) and cached.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from _common import (ROOT, SYN_DIR, environment_info, get_logger, load_all, make_generator, save_json, simulate,
                     uniform_distances, validation_distances)
from channel_model import load_channel_parameters


def build_datasets(profile: str, workers: int | None, log) -> dict:
    sim, prof, _ = load_all(profile)
    p = load_channel_parameters()
    seed, n = int(prof["seed"]), int(prof["n_realizations"])
    info = {"profile": profile, "seed": seed, "n_realizations": n, "env": environment_info(), "datasets": {}}
    for cond in ("los", "nlos"):
        t0 = time.perf_counter()
        r, c = simulate(cond, n, seed, "val", validation_distances(p, cond), workers=workers,
                        cache=f"{profile}_val_{cond}", log=log)
        info["datasets"][f"val_{cond}"] = {"n": len(r), "clusters": len(c), "wall_s": time.perf_counter() - t0,
                                           "mean_time_per_realization_s": float(r.time_s.mean())}
        t0 = time.perf_counter()
        r, c = simulate(cond, n, seed, "pl", uniform_distances(n, seed, cond), workers=workers,
                        cache=f"{profile}_pl_{cond}", log=log)
        info["datasets"][f"pl_{cond}"] = {"n": len(r), "clusters": len(c), "wall_s": time.perf_counter() - t0,
                                          "mean_time_per_realization_s": float(r.time_s.mean())}
        log.info("%s done: %s", cond, info["datasets"])
    save_json(SYN_DIR / f"{profile}_manifest.json", info)
    return info


def example(cond: str, d: float, seed: int, log) -> None:
    sim, _, _ = load_all("smoke")
    gen = make_generator(sim)
    r = gen.generate(d, cond, np.random.default_rng(seed), keep_adps=True)
    out = SYN_DIR / f"example_{cond}_{d:g}m_seed{seed}.npz"
    np.savez_compressed(out, adps=r.adps.astype(np.float32), tap_labels=r.tap_labels, kappa=r.kappa,
                        peak_db=r.peak_db, visible=r.visible, cutoff_db=r.cutoff_db)
    log.info("example %s d=%.1f m: %d clusters (%d visible), %d taps, DS=%.2f m, ASA=%.3f, ASD=%.3f -> %s",
             cond, d, r.clusters.n, r.n_visible, r.metrics["n_taps"], r.metrics["delay_spread_m"],
             r.metrics["doa_spread"], r.metrics["dod_spread"], out.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke", choices=["smoke", "standard", "full"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--example", action="store_true", help="generate and save one ADPS realization only")
    ap.add_argument("--condition", default="los", choices=["los", "nlos"])
    ap.add_argument("--distance", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    log = get_logger("generate_channels")
    if a.example:
        example(a.condition, a.distance, a.seed, log)
    else:
        build_datasets(a.profile, a.workers, log)


if __name__ == "__main__":
    main()
