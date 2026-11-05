"""Calibration of the generator against the paper's channel-level results (Fig. 17).

    python scripts/calibrate_model.py --probe            # sensitivity probe
    python scripts/calibrate_model.py --search           # coordinate search, writes calibrated YAML

Why this exists
---------------
Generated with the Table I parameters exactly as published, the NLoS channels are more
dispersive than both the measured channels and the authors' own synthetic channels in
Fig. 17 (docs/LIMITATIONS.md item 9). Table I describes the *marginal* cluster statistics;
the paper's own toolkit is not available, so a few generator details are unknown.

This script performs an explicit, documented **calibration**: a small number of model
parameters is adjusted so that the generated channel spreads match the paper's synthetic
CDFs (digitised Fig. 17) as closely as possible. The result is stored in a SEPARATE file,
``config/channel_parameters_calibrated.yaml``; the Table I file is never modified, and
every result produced with the calibrated file is labelled "calibrated".

Calibrated knobs (all others stay at Table I values):
  count_scale   multiplies the Gamma scale of the cluster-number law
  rho_scale     multiplies the NCP standard deviation rho of the non-LoS-cluster class
  beta_scale    multiplies the NCP distance slope beta of the non-LoS-cluster class
  alpha_shift   dB added to the NCP intercept alpha of the non-LoS-cluster class
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, simulate, validation_distances  # noqa: E402
from channel_model import load_channel_parameters  # noqa: E402
from channel_model.validation import cvm_distance, load_samples  # noqa: E402

BASE = ROOT / "config" / "channel_parameters.yaml"
OUT = ROOT / "config" / "channel_parameters_calibrated.yaml"
REPORT = ROOT / "results" / "calibration"
METRICS = [("delay", "delay_spread_m", "fig17a_delay_spread"), ("doa", "doa_spread", "fig17b_doa_spread"),
           ("dod", "dod_spread", "fig17c_dod_spread")]
KNOBS = ("count_scale", "rho_scale", "beta_scale", "alpha_shift")
IDENTITY = {"count_scale": 1.0, "rho_scale": 1.0, "beta_scale": 1.0, "alpha_shift": 0.0}
# stage 2 (LoS only): dB added to the NCP intercept of the dominant LoS cluster
LC_KNOB = "lc_alpha_shift"


def modified_yaml(cond: str, knobs: dict, base: dict | None = None) -> dict:
    raw = copy.deepcopy(base or yaml.safe_load(BASE.read_text()))
    k = {**IDENTITY, **knobs}
    cn_key = "los_additional" if cond == "los" else "nlos_minus_one"
    raw["cluster_number"][cn_key]["scale"] = float(raw["cluster_number"][cn_key]["scale"]) * k["count_scale"]
    pw = raw["cluster_power"][cond]
    pw["rho"] = float(pw["rho"]) * k["rho_scale"]
    pw["beta"] = float(pw["beta"]) * k["beta_scale"]
    pw["alpha"] = float(pw["alpha"]) + k["alpha_shift"]
    if cond == "los" and LC_KNOB in k:
        lc = raw["cluster_power"]["los_cluster"]
        lc["alpha"] = float(lc["alpha"]) + float(k[LC_KNOB])
    return raw


def pathloss_error(knobs: dict, n: int, seed: int) -> dict:
    """LoS omni path loss of generated channels vs the paper's synthetic Table II line.

    Returns the alpha-beta fit of n channels at d ~ U(1, 100) m and the RMS dB difference
    between the fitted line and the published one over 1-100 m.
    """
    from _common import uniform_distances
    from channel_model.path_loss import fit_alpha_beta, fit_close_in
    raw = modified_yaml("los", knobs)
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=str(REPORT))
    os.close(fd)
    Path(path).write_text(yaml.safe_dump(raw, sort_keys=False))
    try:
        real, _ = simulate("los", n, seed, "misc", uniform_distances(n, seed, "los"), variant={"path": path})
    finally:
        os.remove(path)
    pub = yaml.safe_load(BASE.read_text())["published_benchmarks"]["path_loss_table_ii"]["alpha_beta"]["los_omni"]
    d, pl = real.d_m.to_numpy(), -real.pg_omni_db.to_numpy()
    ab = fit_alpha_beta(d, pl)
    ci = fit_close_in(d, pl, load_channel_parameters().fc_hz)
    dd = np.logspace(0, 2, 50)
    rms = float(np.sqrt(np.mean((ab.alpha + 10 * ab.beta * np.log10(dd) - pub["alpha"] - 10 * pub["beta"] * np.log10(dd)) ** 2)))
    return {"pl_alpha": ab.alpha, "pl_beta": ab.beta, "pl_sigma": ab.sigma, "pl_n_ci": ci.n_exp, "pl_rms_db": rms}


def evaluate(cond: str, knobs: dict, reps: int, seed: int, log=None) -> dict:
    """Generate reps x (measured link distances) channels and score them against Fig. 17."""
    raw = modified_yaml(cond, knobs)
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=str(REPORT))
    os.close(fd)
    Path(path).write_text(yaml.safe_dump(raw, sort_keys=False))
    try:
        p = load_channel_parameters()
        d = validation_distances(p, cond)
        real, _ = simulate(cond, reps * d.size, seed, "misc", d, variant={"path": path}, log=None)
    finally:
        os.remove(path)
    out = {"cond": cond, **{k: float(v) for k, v in {**IDENTITY, **knobs}.items()},
           "n": int(len(real)), "mean_visible": float(real.n_visible.mean())}
    score = 0.0
    for key, col, fig in METRICS:
        ours = real[col].to_numpy()
        ours = ours[np.isfinite(ours)]
        syn = load_samples(fig, f"{cond}_paper_synthetic")
        meas = load_samples(fig, f"{cond}_measured")
        out[f"cvm_vs_paper_syn_{key}"] = cvm_distance(ours, syn)
        out[f"cvm_vs_measured_{key}"] = cvm_distance(ours, meas)
        out[f"median_{key}"] = float(np.median(ours))
        score += out[f"cvm_vs_paper_syn_{key}"]
    out["score"] = score
    if log:
        log.info("%s %s -> score %.4f | medians DS %.2f m ASA %.3f ASD %.3f | visible %.1f", cond,
                 {k: round(out[k], 3) for k in KNOBS}, score, out["median_delay"], out["median_doa"],
                 out["median_dod"], out["mean_visible"])
    return out


def probe(log, reps: int, seed: int) -> list[dict]:
    rows = []
    for cond, knobs in [("nlos", {}), ("nlos", {"rho_scale": 1.4}), ("nlos", {"beta_scale": 1.6}),
                        ("nlos", {"count_scale": 0.4}), ("los", {}), ("los", {"alpha_shift": 4.0}),
                        ("los", {"count_scale": 1.6})]:
        rows.append(evaluate(cond, knobs, reps, seed, log))
    return rows


def search(log, cond: str, reps: int, seed: int, grid: dict) -> tuple[dict, list[dict]]:
    """Coordinate search over the knob grid: sweep one knob at a time, keep the best, repeat."""
    best_k = dict(IDENTITY)
    best = evaluate(cond, best_k, reps, seed, log)
    hist = [best]
    for sweep in range(2):
        improved = False
        for knob, values in grid.items():
            for v in values:
                if np.isclose(v, best_k[knob]):
                    continue
                cand = {**best_k, knob: v}
                r = evaluate(cond, cand, reps, seed, log)
                hist.append(r)
                if r["score"] < best["score"]:
                    best, best_k, improved = r, cand, True
        log.info("%s sweep %d best %s score %.4f", cond, sweep + 1, best_k, best["score"])
        if not improved:
            break
    return best_k, hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--stage2", action="store_true",
                    help="LoS: also tune the LoS-cluster NCP intercept against the Table II path-loss line")
    ap.add_argument("--stage3", action="store_true",
                    help="NLoS: set the absolute level so that the close-in exponent matches Table II")
    ap.add_argument("--reps", type=int, default=4, help="channels per measured link distance")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    log = get_logger("calibrate_model")
    if a.probe:
        rows = probe(log, a.reps, a.seed)
        (REPORT / "probe.json").write_text(json.dumps(rows, indent=2))
    if a.search:
        grids = {
            "nlos": {"rho_scale": [1.2, 1.4, 1.7, 2.0, 2.4], "count_scale": [0.25, 0.4, 0.6, 1.0],
                     "beta_scale": [1.0, 1.4, 1.8], "alpha_shift": [0.0]},
            "los": {"alpha_shift": [0.0, 2.0, 4.0, 6.0], "count_scale": [1.0, 1.3, 1.6], "rho_scale": [0.8, 1.0, 1.2],
                    "beta_scale": [1.0]},
        }
        result, hist = {}, []
        for cond in ("los", "nlos"):
            k, h = search(log, cond, a.reps, a.seed, grids[cond])
            result[cond] = k
            hist.extend(h)
        raw = yaml.safe_load(BASE.read_text())
        for cond in ("los", "nlos"):
            raw = modified_yaml(cond, result[cond], raw)
        raw["calibration"] = {
            "note": "CALIBRATED variant of Table I (scripts/calibrate_model.py). Not the published parameters. "
                    "Knobs fitted so that generated channel spreads match the paper's synthetic Fig. 17 CDFs.",
            "knobs": result, "reps_per_distance": a.reps, "seed": a.seed,
        }
        OUT.write_text("# AUTO-GENERATED by scripts/calibrate_model.py - calibrated variant, NOT Table I\n"
                       + yaml.safe_dump(raw, sort_keys=False))
        (REPORT / "search_history.json").write_text(json.dumps(hist, indent=2))
        (REPORT / "calibrated_knobs.json").write_text(json.dumps(result, indent=2))
        log.info("calibrated knobs: %s -> %s", result, OUT)
    if a.stage2:
        result = json.loads((REPORT / "calibrated_knobs.json").read_text())
        base_k = dict(result["los"])
        best, best_k, hist = None, None, []
        for lc in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0):
            for ash in sorted({base_k["alpha_shift"], base_k["alpha_shift"] + 1.0, base_k["alpha_shift"] + 2.0}):
                k = {**base_k, "alpha_shift": ash, LC_KNOB: lc}
                r = evaluate("los", k, a.reps, a.seed)
                r.update(pathloss_error(k, 150, a.seed))
                r[LC_KNOB] = lc
                # spreads (CvM vs paper synthetic) + path-loss line mismatch, both of order 1e-2
                r["score2"] = r["score"] + 0.01 * r["pl_rms_db"] ** 2
                hist.append(r)
                log.info("stage2 lc %+.1f ash %.1f -> CvM sum %.4f, PL alpha %.2f beta %.2f rms %.2f dB -> score2 %.4f",
                         lc, ash, r["score"], r["pl_alpha"], r["pl_beta"], r["pl_rms_db"], r["score2"])
                if best is None or r["score2"] < best["score2"]:
                    best, best_k = r, k
        result["los"] = best_k
        raw = yaml.safe_load(BASE.read_text())
        for cond in ("los", "nlos"):
            raw = modified_yaml(cond, result[cond], raw)
        raw["calibration"] = {
            "note": "CALIBRATED variant of Table I (scripts/calibrate_model.py). Not the published parameters. "
                    "Knobs fitted so that generated channel spreads match the paper's synthetic Fig. 17 CDFs "
                    "(and, for LoS, the paper's synthetic Table II path-loss line).",
            "knobs": result, "reps_per_distance": a.reps, "seed": a.seed,
        }
        OUT.write_text("# AUTO-GENERATED by scripts/calibrate_model.py - calibrated variant, NOT Table I\n"
                       + yaml.safe_dump(raw, sort_keys=False))
        (REPORT / "stage2_history.json").write_text(json.dumps(hist, indent=2))
        (REPORT / "calibrated_knobs.json").write_text(json.dumps(result, indent=2))
        log.info("stage 2 knobs: %s -> %s", result, OUT)
    if a.stage3:
        # Absolute level.  The 35 dB cut-off is relative to the strongest tap, so adding the same number
        # of dB to the NCP intercept of EVERY cluster of a condition (for LoS: the LoS cluster too) changes
        # only the absolute level of the channel (path loss), not its shape (spreads).  The level is set so
        # that the close-in exponent equals the paper's synthetic Table II value; the spreads are
        # re-evaluated to show they are unchanged.  For LoS this replaces the stage-2 LoS-cluster-only
        # shift (which changed the shape and widened the DoD spread): stage-1 shape + common level.
        from _common import uniform_distances
        from channel_model.path_loss import fit_close_in
        result = json.loads((REPORT / "calibrated_knobs.json").read_text())
        fc = load_channel_parameters().fc_hz
        ci_pub = yaml.safe_load(BASE.read_text())["published_benchmarks"]["path_loss_table_ii"]["close_in"]

        def ci_fit(cond, knobs):
            raw = modified_yaml(cond, knobs)
            fd, path = tempfile.mkstemp(suffix=".yaml", dir=str(REPORT))
            os.close(fd)
            Path(path).write_text(yaml.safe_dump(raw, sort_keys=False))
            try:
                real, _ = simulate(cond, 150, a.seed, "misc", uniform_distances(150, a.seed, cond), variant={"path": path})
            finally:
                os.remove(path)
            d, pl = real.d_m.to_numpy(), -real.pg_omni_db.to_numpy()
            return d, fit_close_in(d, pl, fc).n_exp

        report = {}
        for cond in ("los", "nlos"):
            k0 = dict(result[cond])
            if cond == "los":
                k0[LC_KNOB] = 0.0      # undo stage 2: back to the stage-1 shape (LoS cluster as in Table I)
            n_target = ci_pub[f"{cond}_omni"]["n_fit"]
            before = evaluate(cond, k0, a.reps, a.seed, log)
            d, n0 = ci_fit(cond, k0)
            x = 10 * np.log10(d)
            c = (n_target - n0) * np.sum(x * x) / np.sum(x)   # dB to add to the path loss (least squares)
            k1 = {**k0, "alpha_shift": float(round(k0["alpha_shift"] - c, 2))}
            if cond == "los":
                k1[LC_KNOB] = float(round(k0[LC_KNOB] - c, 2))
            after = evaluate(cond, k1, a.reps, a.seed, log)
            _, n1 = ci_fit(cond, k1)
            log.info("stage3 %s: n_CI %.3f -> %.3f (target %.2f), common level %+.2f dB; CvM sum %.4f -> %.4f",
                     cond, n0, n1, n_target, -c, before["score"], after["score"])
            result[cond] = k1
            report[cond] = {"n_ci_before": n0, "n_ci_after": n1, "n_target": n_target, "level_db": -c,
                            "before": before, "after": after}
        raw = yaml.safe_load(BASE.read_text())
        for cond in ("los", "nlos"):
            raw = modified_yaml(cond, result[cond], raw)
        raw["calibration"] = {
            "note": "CALIBRATED variant of Table I (scripts/calibrate_model.py). Not the published parameters. "
                    "Cluster-power knobs fitted so that generated channel spreads match the paper's synthetic Fig. 17 "
                    "CDFs; then one common level per condition (all clusters, same dB) set so that the close-in "
                    "path-loss exponent equals the paper's synthetic Table II value.",
            "knobs": result, "reps_per_distance": a.reps, "seed": a.seed,
        }
        OUT.write_text("# AUTO-GENERATED by scripts/calibrate_model.py - calibrated variant, NOT Table I\n"
                       + yaml.safe_dump(raw, sort_keys=False))
        (REPORT / "stage3.json").write_text(json.dumps(report, indent=2))
        (REPORT / "calibrated_knobs.json").write_text(json.dumps(result, indent=2))
        log.info("stage 3 knobs: %s -> %s", result, OUT)


if __name__ == "__main__":
    main()
