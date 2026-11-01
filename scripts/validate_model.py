"""Model validation (Secs. VII-VIII): CvM distances (Fig. 17) and path-loss models (Table II).

    python scripts/validate_model.py --profile smoke|standard|full

Inputs
  * synthetic data sets from scripts/generate_channels.py (SIMULATED with Table I);
  * digitised CDFs of the published Fig. 17 (measured and the authors' synthetic
    channel spreads), produced by scripts/digitize_published_figures.py.  These are
    reconstructions of plotted step CDFs, not raw measurement data.
Outputs
  results/tables/cvm_validation.csv, results/tables/pathloss_table2.csv,
  results/tables/pathloss_repeats_<profile>.csv, results/tables/validation_summary.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from _common import ROOT, TAB_DIR, get_logger, load_all, save_json, simulate, uniform_distances, validation_distances
from channel_model import load_channel_parameters
from channel_model.path_loss import ci_overlap, fit_alpha_beta, fit_close_in
from channel_model.validation import bootstrap_cvm_null, cvm_distance, cvm_pvalue, digitized_available, load_samples

METRICS = [("delay", "delay_spread_m", "fig17a_delay_spread"), ("doa", "doa_spread", "fig17b_doa_spread"),
           ("dod", "dod_spread", "fig17c_dod_spread")]


def bench() -> dict:
    return yaml.safe_load((ROOT / "config" / "channel_parameters.yaml").read_text())["published_benchmarks"]


def cvm_table(profile: str, log) -> pd.DataFrame:
    sim, prof, _ = load_all(profile)
    p = load_channel_parameters()
    pub = bench()["cvm"]
    rng = np.random.default_rng(int(prof["seed"]) + 17)
    rows = []
    for cond in ("los", "nlos"):
        real, _ = simulate(cond, int(prof["n_realizations"]), int(prof["seed"]), "val", validation_distances(p, cond),
                           cache=f"{profile}_val_{cond}", log=log)
        for key, col, fig in METRICS:
            ours = real[col].to_numpy()
            ours = ours[np.isfinite(ours)]
            row = {"metric": key, "condition": cond, "n_ours": ours.size, "published_cvm": pub[key][cond]}
            if digitized_available():
                meas = load_samples(fig, f"{cond}_measured")
                paper_syn = load_samples(fig, f"{cond}_paper_synthetic")
                null = bootstrap_cvm_null(meas.size, ours.size, 2000, rng)
                row.update({
                    "n_measured_digitized": meas.size, "n_paper_synthetic_digitized": paper_syn.size,
                    "recomputed_paper_cvm": cvm_distance(meas, paper_syn),
                    "ours_vs_measured_cvm": cvm_distance(ours, meas),
                    "ours_vs_paper_synthetic_cvm": cvm_distance(ours, paper_syn),
                    "null_cvm_median": float(np.median(null)), "null_cvm_95": float(np.quantile(null, 0.95)),
                    "p_value_ours_vs_measured": cvm_pvalue(ours, meas),
                    "median_measured": float(np.median(meas)), "median_paper_synthetic": float(np.median(paper_syn)),
                })
                row["diff_ours_minus_published"] = row["ours_vs_measured_cvm"] - row["published_cvm"]
                row["status"] = ("consistent with measured (within sampling null 95%)"
                                 if row["ours_vs_measured_cvm"] <= row["null_cvm_95"] else
                                 "deviates from measured beyond sampling noise")
            else:
                row["status"] = "not computed: digitized curves unavailable"
            row["median_ours"] = float(np.median(ours))
            row["classification"] = "Statistical reproduction using published parameters (vs digitized CDFs)"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "cvm_validation.csv", index=False, float_format="%.5g")
    df.to_csv(TAB_DIR / f"cvm_validation_{profile}.csv", index=False, float_format="%.5g")
    log.info("CvM table:\n%s", df[["metric", "condition", "published_cvm", "recomputed_paper_cvm",
                                   "ours_vs_measured_cvm", "null_cvm_95", "median_ours", "median_measured"]].to_string())
    return df


def pathloss(profile: str, log) -> pd.DataFrame:
    sim, prof, _ = load_all(profile)
    p = load_channel_parameters()
    pub = bench()["path_loss_table_ii"]
    n_ch = int(prof.get("n_pathloss_channels", 50))
    n_rep = int(prof.get("n_pathloss_repeats", 20))
    rng = np.random.default_rng(int(prof["seed"]) + 29)
    rows, reps = [], []
    for cond in ("los", "nlos"):
        n = int(prof["n_realizations"])
        real, _ = simulate(cond, n, int(prof["seed"]), "pl", uniform_distances(n, int(prof["seed"]), cond),
                           cache=f"{profile}_pl_{cond}", log=log)
        for mode, col in (("omni", "pg_omni_db"), ("maxdir", "pg_maxdir_db")):
            d_all = real["d_m"].to_numpy()
            pl_all = -real[col].to_numpy()
            key = f"{cond}_{mode}"
            for r in range(n_rep):
                # experiment of Sec. VIII: 50 channels at distances ~ U(<=100 m); repeats are drawn
                # without replacement from the generated pool (pool >> 50, see docs/METHODOLOGY.md)
                idx = rng.choice(d_all.size, size=min(n_ch, d_all.size), replace=False)
                ab = fit_alpha_beta(d_all[idx], pl_all[idx])
                ci = fit_close_in(d_all[idx], pl_all[idx], p.fc_hz)
                reps.append({"case": key, "repeat": r, "alpha": ab.alpha, "beta": ab.beta, "sigma_ab": ab.sigma,
                             "n_cim": ci.n_exp, "sigma_cim": ci.sigma,
                             "alpha_ci_lo": ab.alpha_ci[0], "alpha_ci_hi": ab.alpha_ci[1],
                             "beta_ci_lo": ab.beta_ci[0], "beta_ci_hi": ab.beta_ci[1],
                             "sigma_ab_ci_lo": ab.sigma_ci[0], "sigma_ab_ci_hi": ab.sigma_ci[1],
                             "n_cim_ci_lo": ci.n_ci[0], "n_cim_ci_hi": ci.n_ci[1],
                             "sigma_cim_ci_lo": ci.sigma_ci[0], "sigma_cim_ci_hi": ci.sigma_ci[1]})
            rr = pd.DataFrame([x for x in reps if x["case"] == key])
            first = rr.iloc[0]
            ab_pub, ci_pub = pub["alpha_beta"][key], pub["close_in"][key]
            for par, ours_col, lo, hi, pub_val, pub_ci, ref in (
                    ("alpha", "alpha", "alpha_ci_lo", "alpha_ci_hi", ab_pub["alpha"], ab_pub["alpha_ci"], ab_pub["alpha_ref"]),
                    ("beta", "beta", "beta_ci_lo", "beta_ci_hi", ab_pub["beta"], ab_pub["beta_ci"], ab_pub["beta_ref"]),
                    ("sigma_ab", "sigma_ab", "sigma_ab_ci_lo", "sigma_ab_ci_hi", ab_pub["sigma"], ab_pub["sigma_ci"], ab_pub["sigma_ref"]),
                    ("n_cim", "n_cim", "n_cim_ci_lo", "n_cim_ci_hi", ci_pub["n_fit"], ci_pub["n_fit_ci"], ci_pub["n_mea"]),
                    ("sigma_cim", "sigma_cim", "sigma_cim_ci_lo", "sigma_cim_ci_hi", ci_pub["sigma_fit"], ci_pub["sigma_fit_ci"], ci_pub["sigma_mea"])):
                vals = rr[ours_col].to_numpy()
                ours_ci = (float(first[lo]), float(first[hi]))
                rows.append({"case": key, "parameter": par, "measured_ref_[21]": ref, "published_synthetic": pub_val,
                             "published_ci_lo": pub_ci[0], "published_ci_hi": pub_ci[1],
                             "ours_first_50": float(first[ours_col]), "ours_first_ci_lo": ours_ci[0],
                             "ours_first_ci_hi": ours_ci[1], "ours_median_over_repeats": float(np.median(vals)),
                             "ours_p2.5": float(np.quantile(vals, 0.025)), "ours_p97.5": float(np.quantile(vals, 0.975)),
                             "n_repeats": n_rep, "first_ci_overlaps_published_ci": ci_overlap(ours_ci, tuple(pub_ci)),
                             "share_repeats_ci_overlap_published": float(np.mean(
                                 [ci_overlap((a, b), tuple(pub_ci)) for a, b in zip(rr[lo], rr[hi])]))})
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "pathloss_table2.csv", index=False, float_format="%.4g")
    pd.DataFrame(reps).to_csv(TAB_DIR / f"pathloss_repeats_{profile}.csv", index=False, float_format="%.5g")
    log.info("Path loss (Table II):\n%s", df[["case", "parameter", "measured_ref_[21]", "published_synthetic",
                                              "ours_median_over_repeats", "ours_p2.5", "ours_p97.5",
                                              "share_repeats_ci_overlap_published"]].to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke", choices=["smoke", "standard", "full"])
    a = ap.parse_args()
    log = get_logger("validate_model")
    cvm = cvm_table(a.profile, log)
    pl = pathloss(a.profile, log)
    save_json(TAB_DIR / "validation_summary.json", {
        "profile": a.profile,
        "cvm": cvm.to_dict(orient="records"),
        "pathloss": pl.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
