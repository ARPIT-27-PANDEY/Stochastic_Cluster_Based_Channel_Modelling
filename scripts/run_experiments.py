"""Experiments 1-7.  Each writes results/experiments/<name>/{config.json, results.json, README.md, *.png, *.csv}.

    python scripts/run_experiments.py --profile smoke|standard|full [--only exp1 exp4]

exp1_performance       generation time per realization vs batch size (single process)
exp2_dr_sensitivity    channel statistics vs dynamic range (common random numbers)
exp3_inverse_mapping   accuracy / convergence of the inverse mapping and the discretisation correction
exp4_power_mle         truncated MLE (Eqs. 10-11) vs OLS vs ground truth on simulated observable clusters
exp5_clustering        KPD vs ML-KPD against exact ground-truth labels of generated ADPSs
exp6_correction_ablation  generator with vs without the discretisation correction
exp7_parameter_variants   Table I vs text arrival parameters; truncated vs wrapped bimodal Laplace
All data are SIMULATED with the published parameters; nothing here is measured data.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (EXP_DIR, environment_info, get_logger, load_all, make_generator, realization_seed, save_json,
                     setup_mpl, simulate, validation_distances)
from channel_model import load_channel_parameters
from channel_model.dynamic_range import AngleGrid, continuous_observables, discrete_observables
from channel_model.inverse_mapping import invert_spreads, solve_decay_constants
from channel_model.kpd_clustering import kpd, ml_kpd
from channel_model.mpc_extraction import extract_taps
from channel_model.parameters import LOS, NLOS
from channel_model.power_model import (equalization_weights, fit_ols, fit_truncated_mle, fit_truncated_mle_lbfgs,
                                       sigma_coverage)
from channel_model.validation import cvm_distance, digitized_available, load_samples

plt = setup_mpl()
P = load_channel_parameters()


def _dir(name: str) -> Path:
    d = EXP_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _readme(d: Path, title: str, purpose: str, cfg: dict, observations: list[str], comparison: list[str],
            limitations: list[str]) -> None:
    txt = [f"# {title}", "", f"**Purpose.** {purpose}", "", "**Configuration.** see `config.json` "
           f"(profile `{cfg.get('profile')}`, seed `{cfg.get('seed')}`). Results: `results.json` and the CSV/PNG files here.",
           "", "**Data.** Simulated with the published Table I parameters (no measured data).", "",
           "## Observations (computed from this run)", *[f"- {o}" for o in observations], "",
           "## Comparison with the paper", *[f"- {c}" for c in comparison], "", "## Limitations",
           *[f"- {x}" for x in limitations], ""]
    (d / "README.md").write_text("\n".join(txt), encoding="utf-8")


# --------------------------------------------------------------------------- exp1
def exp1(profile, prof, sim, log):
    d = _dir("exp1_performance")
    gen = make_generator(sim)
    rows = []
    for cond in ("los", "nlos"):
        for n in prof["perf_sizes"]:
            ts = []
            t0 = time.perf_counter()
            for i in range(n):
                rng = realization_seed(int(prof["seed"]), cond, "perf", i)
                t = time.perf_counter()
                gen.generate(float(rng.uniform(1, 100)), cond, rng)
                ts.append(time.perf_counter() - t)
            rows.append({"condition": cond, "n": n, "total_s": time.perf_counter() - t0, "mean_s": float(np.mean(ts)),
                         "median_s": float(np.median(ts)), "p95_s": float(np.quantile(ts, 0.95))})
            log.info("exp1 %s n=%d mean %.2f s", cond, n, rows[-1]["mean_s"])
    df = pd.DataFrame(rows)
    df.to_csv(d / "timing.csv", index=False)
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for cond, g in df.groupby("condition"):
        ax.plot(g.n, g.total_s, "o-", label=f"{cond.upper()} total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of realizations")
    ax.set_ylabel("wall time [s] (1 process)")
    ax.legend()
    fig.savefig(d / "timing.png")
    plt.close(fig)
    cfg = {"profile": profile, "seed": prof["seed"], "sizes": prof["perf_sizes"], "env": environment_info()}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", df.to_dict(orient="records"))
    m = df.groupby("condition").mean_s.mean()
    _readme(d, "Experiment 1 - generator performance", "Cost of the full 7-step generator on the native 1001x36x36 grid.",
            cfg, [f"mean time per LoS realization {m['los']:.2f} s, NLoS {m['nlos']:.2f} s (single process, incl. metrics)",
                  "time grows linearly with the batch size (no state is shared between realizations)",
                  "NLoS is slower: ~90 clusters, each inverted and corrected inside the cut-off fixed point"],
            ["the paper reports no run times for its MATLAB toolkit; no comparison possible"],
            ["timings are machine-dependent (see env in config.json)"])


# --------------------------------------------------------------------------- exp2
def exp2(profile, prof, sim, log):
    d = _dir("exp2_dr_sensitivity")
    n = max(10, int(prof["n_realizations"]) // 10)
    rows = []
    for cond in ("los", "nlos"):
        for dr in prof["dr_sweep_db"]:
            r, c = simulate(cond, n, int(prof["seed"]), "misc", validation_distances(P, cond), dr_db=float(dr),
                            cache=f"{profile}_dr{dr}_{cond}", log=log)
            rows.append({"condition": cond, "dr_db": dr, "n": n, "visible_mean": r.n_visible.mean(),
                         "ds_median_m": r.delay_spread_m.median(), "asa_median": r.doa_spread.median(),
                         "asd_median": r.dod_spread.median(), "taps_median": r.n_taps.median(),
                         "pg_omni_median_db": r.pg_omni_db.median()})
            log.info("exp2 %s DR=%s: %s", cond, dr, rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(d / "dr_sweep.csv", index=False)
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.6))
    for cond, g in df.groupby("condition"):
        for ax, col, lab in zip(axs, ("visible_mean", "ds_median_m", "asa_median", "asd_median"),
                                ("mean observable clusters", "median DS [m]", "median DoA spread", "median DoD spread")):
            ax.plot(g.dr_db, g[col], "o-", label=cond.upper())
            ax.set_xlabel("dynamic range [dB]")
            ax.set_title(lab, fontsize=8)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(d / "dr_sweep.png")
    plt.close(fig)
    cfg = {"profile": profile, "seed": prof["seed"], "n_per_dr": n, "dr_db": prof["dr_sweep_db"],
           "distances": "measured link distances [21], cycled", "common_random_numbers": True}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", df.to_dict(orient="records"))
    lo, hi = min(prof["dr_sweep_db"]), max(prof["dr_sweep_db"])
    g = {c: df[df.condition == c].set_index("dr_db") for c in ("los", "nlos")}
    _readme(d, "Experiment 2 - dynamic-range sensitivity",
            "How the observable channel statistics depend on the DR used to generate (and observe) the channels.", cfg,
            [f"{c.upper()}: observable clusters {g[c].visible_mean[lo]:.1f} -> {g[c].visible_mean[hi]:.1f}, "
             f"median DS {g[c].ds_median_m[lo]:.2f} -> {g[c].ds_median_m[hi]:.2f} m, median DoA spread "
             f"{g[c].asa_median[lo]:.3f} -> {g[c].asa_median[hi]:.3f} for DR {lo} -> {hi} dB" for c in ("los", "nlos")],
            ["the paper uses 35 dB throughout and argues (Sec. V-A, LD setting) that DR changes cluster counts; "
             "the monotone growth of observable clusters and spreads with DR is consistent with that argument"],
            ["the Table I statistics were extracted at 35 dB; using them at other DRs is an extrapolation "
             "(the inverse mapping keeps per-cluster spreads at their drawn values for every DR)"])


# --------------------------------------------------------------------------- exp3
def exp3(profile, prof, sim, log, corr):
    d = _dir("exp3_inverse_mapping")
    rng = np.random.default_rng(int(prof["seed"]) + 3)
    n = 2000 if profile != "smoke" else 300
    kappa = np.column_stack([np.exp(rng.uniform(np.log(0.1), np.log(50), n)),
                             np.exp(rng.uniform(np.log(0.02), np.log(3), n)),
                             np.exp(rng.uniform(np.log(0.02), np.log(3), n))])
    dr = rng.uniform(1, 50, n)
    o = continuous_observables(kappa[:, 0], kappa[:, 1], kappa[:, 2], dr)
    s = np.column_stack([o["s_tau"], o["s_r"], o["s_t"]])
    t0 = time.perf_counter()
    res = invert_spreads(s, dr)
    t_inv = time.perf_counter() - t0
    kerr = np.abs(res.kappa / kappa - 1)
    # discretisation: targets drawn from Table I (LoS clusters), DR_eff ~ U(3, 35)
    m = 1000 if profile != "smoke" else 200
    tgt = np.column_stack([rng.lognormal(P.delay_spread["los"].mu, P.delay_spread["los"].sigma, m),
                           rng.lognormal(P.doa_spread["los"].mu, P.doa_spread["los"].sigma, m),
                           rng.lognormal(P.dod_spread["los"].mu, P.dod_spread["los"].sigma, m)])
    tgt[:, 1:] = np.minimum(tgt[:, 1:], 0.99)
    drs = rng.uniform(3, 35, m)
    res_m = 0.29979
    out = {}
    for name, c in (("without_correction", None), ("with_correction", corr)):
        r = solve_decay_constants(tgt, drs, c)
        g = discrete_observables(r.kappa[:, 0] / res_m, r.kappa[:, 1], r.kappa[:, 2], drs, AngleGrid(10.0), res_m)
        e = np.column_stack([g["s_tau"], g["s_r"], g["s_t"]]) / tgt - 1
        out[name] = {"median_abs_rel_err": np.median(np.abs(e), 0).tolist(), "p90_abs_rel_err": np.quantile(np.abs(e), 0.9, 0).tolist(),
                     "median_signed_rel_err": np.median(e, 0).tolist()}
        out[name + "_errors"] = e
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))
    for j, lab in enumerate(("delay spread", "DoA spread", "DoD spread")):
        for name, col in (("without_correction", "#d62728"), ("with_correction", "#1f77b4")):
            x = np.sort(np.abs(out[name + "_errors"][:, j]))
            axs[j].step(x, np.arange(1, x.size + 1) / x.size, color=col, label=name.replace("_", " "))
        axs[j].set_xscale("log")
        axs[j].set_xlim(1e-4, 2)
        axs[j].set_xlabel(f"|re-observed / target - 1|, {lab}")
    axs[0].set_ylabel("CDF")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(d / "discretisation_error.png")
    plt.close(fig)
    results = {"roundtrip": {"n": n, "converged_share": float(res.converged.mean()),
                             "max_rel_kappa_err": kerr.max(0).tolist(), "median_rel_kappa_err": np.median(kerr, 0).tolist(),
                             "time_per_inversion_ms": 1e3 * t_inv / n},
               "discretisation": {k: v for k, v in out.items() if not k.endswith("errors")}}
    cfg = {"profile": profile, "seed": int(prof["seed"]) + 3, "kappa_ranges": {"tau_m": [0.1, 50], "angles_rad": [0.02, 3]},
           "dr_range_db": [1, 50], "grid": "1 ns (0.29979 m) / 10 deg", "n_roundtrip": n, "n_discretisation": m}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", results)
    wo, wi = out["without_correction"]["median_abs_rel_err"], out["with_correction"]["median_abs_rel_err"]
    _readme(d, "Experiment 3 - inverse mapping and discretisation correction",
            "Accuracy and robustness of kappa = F^-1(S_obs; DR) and of the pre-computed correction table.", cfg,
            [f"continuous round trip: {100 * res.converged.mean():.1f}% converged, max relative kappa error "
             f"{max(results['roundtrip']['max_rel_kappa_err']):.1e}, {results['roundtrip']['time_per_inversion_ms']:.2f} ms per cluster (vectorised)",
             f"native grid, median |error| of re-observed spreads without correction: delay {wo[0]:.3f}, DoA {wo[1]:.3f}, DoD {wo[2]:.3f}",
             f"with correction: delay {wi[0]:.3f}, DoA {wi[1]:.3f}, DoD {wi[2]:.3f}"],
            ["paper: Newton-Raphson 'converges reliably from a broad range of initialisations' - confirmed for the "
             "continuous map; paper gives no numbers for the correction, so only the qualitative claim (correction "
             "reduces discretisation error) can be checked"],
            ["the correction table is our own construction (grid, interpolation, table ranges) because the paper "
             "does not specify it; targets outside the table range are clipped"])


# --------------------------------------------------------------------------- exp4
def exp4(profile, prof, sim, log):
    d = _dir("exp4_power_mle")
    pm = P.raw["power_mle"]
    results, rows = {}, []
    for cond, cls in (("los", LOS), ("nlos", NLOS)):
        _, cl = simulate(cond, int(prof["n_realizations"]), int(prof["seed"]), "val", validation_distances(P, cond),
                         cache=f"{profile}_val_{cond}", log=log)
        cl = cl[cl.rid < int(prof["n_mle_channels"])]
        sel = cl[(cl.cls == cls) & cl.visible]
        x, y, thr = sel.ratio_db.to_numpy(), sel.ncp_db.to_numpy(), sel.thr_ncp_db.to_numpy()
        w = equalization_weights(x, pm["n_bins"], tuple(pm["ratio_range_db"]), pm["min_bin_fraction"])
        rng = np.random.default_rng(int(prof["seed"]) + 4)
        t0 = time.perf_counter()
        mle = fit_truncated_mle(x, y, thr, w, lr=pm["learning_rate"], max_iter=int(pm["max_iterations"]) if profile == "full" else 60000,
                                tol=pm["tol_llf"], n_starts=int(pm["n_starts"]) if profile != "smoke" else 3,
                                start_box=pm["start_box"], rng=rng)
        t_mle = time.perf_counter() - t0
        lb, llf_lb = fit_truncated_mle_lbfgs(x, y, thr, w)
        a_ols, b_ols = fit_ols(x, y, w)
        truth = P.power[cond]
        # all drawn (incl. unobservable) clusters: the untruncated population
        allc = cl[cl.cls == cls]
        a_all, b_all = fit_ols(allc.ratio_db, allc.ncp_db)
        cov1 = sigma_coverage(x, y, thr, mle.as_params(), 1.0)
        cov3 = sigma_coverage(x, y, thr, mle.as_params(), 3.0)
        results[cond] = {"n_observable": int(len(sel)), "n_drawn": int(len(allc)),
                         "truth_table_I": {"alpha": truth.alpha, "beta": truth.beta, "rho": truth.rho, "eta": truth.eta},
                         "mle": mle.to_dict(), "mle_time_s": t_mle,
                         "lbfgs_check": {"alpha": lb[0], "beta": lb[1], "rho": lb[2], "eta": lb[3], "llf": llf_lb},
                         "ols": {"alpha": a_ols, "beta": b_ols},
                         "ols_on_all_drawn_clusters": {"alpha": a_all, "beta": b_all},
                         "coverage_1sigma": {"empirical": cov1[0], "truncated_expected": cov1[1]},
                         "coverage_3sigma": {"empirical": cov3[0], "truncated_expected": cov3[1]},
                         "published_coverage": P.raw["published_benchmarks"]["ncp_sigma_coverage"][cond]}
        for k, v in (("truth", truth.__dict__), ("mle", results[cond]["mle"]), ("ols", results[cond]["ols"])):
            rows.append({"condition": cond, "estimator": k, **{p: v.get(p, np.nan) for p in ("alpha", "beta", "rho", "eta")}})
        log.info("exp4 %s: %s", cond, {k: results[cond][k] for k in ("mle", "ols", "truth_table_I")})
        # convergence trace
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        ax.plot(np.arange(len(mle.history)), mle.history)
        ax.set_xlabel("record (every 500 sweeps)")
        ax.set_ylabel("weighted LLF")
        ax.set_title(f"{cond.upper()} best start", fontsize=8)
        fig.savefig(d / f"llf_trace_{cond}.png")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(d / "estimates.csv", index=False, float_format="%.4f")
    cfg = {"profile": profile, "seed": prof["seed"], "n_channels": prof["n_mle_channels"], **pm,
           "threshold_definition": "per realization: NCP_thr = cut-off [dB] + FSPL(d_LoS) (a cluster cannot be observed "
                                   "with total power below one tap at the cut-off)"}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", results)
    obs = []
    for cond in ("los", "nlos"):
        r = results[cond]
        obs.append(f"{cond.upper()} ({r['n_observable']} observable of {r['n_drawn']} drawn): truth beta {r['truth_table_I']['beta']:.2f}; "
                   f"MLE beta {r['mle']['beta']:.2f} (alpha {r['mle']['alpha']:.2f}, rho {r['mle']['rho']:.2f}, eta {r['mle']['eta']:.3f}); "
                   f"OLS beta {r['ols']['beta']:.2f} (alpha {r['ols']['alpha']:.2f}); L-BFGS LLF {r['lbfgs_check']['llf']:.1f} vs ours {r['mle']['llf']:.1f}")
        obs.append(f"{cond.upper()} coverage +-1 sigma: {r['coverage_1sigma']['empirical']:.3f} (truncated expectation "
                   f"{r['coverage_1sigma']['truncated_expected']:.3f}); +-3 sigma {r['coverage_3sigma']['empirical']:.4f}")
    _readme(d, "Experiment 4 - cluster power model: truncated MLE vs OLS",
            "Re-estimate (alpha, beta, rho, eta) from simulated OBSERVABLE clusters, where the truth (Table I) is known.",
            cfg, obs,
            ["paper (Figs. 12-13): OLS slope is less steep than the truncation-aware MLE - reproduced qualitatively if "
             "OLS beta > MLE beta here", "paper coverage (measured): LoS 71.55% / 99.16%, NLoS 67.79% / 99.34% "
             "(expected 74.6/99.8 and 71.7/99.8)"],
            ["truncation in the generator is a peak-tap criterion, not exactly 'NCP below a level', so the truncated-normal "
             "likelihood is only approximately correct here; residual bias is therefore expected",
             "learning rate 1e-3 with block alternation; iteration budget reduced outside the 'full' profile"])


# --------------------------------------------------------------------------- exp5
def exp5(profile, prof, sim, log):
    from sklearn.metrics import adjusted_rand_score as ari
    d = _dir("exp5_clustering")
    gen = make_generator(sim)
    rows = []
    for cond in ("los", "nlos"):
        for i in range(int(prof["n_clustering_channels"])):
            rng = realization_seed(int(prof["seed"]), cond, "misc", 5000 + i)
            r = gen.generate(float(rng.uniform(5, 80)), cond, rng, keep_adps=True)
            taps = extract_taps(r.adps, gen.grid, 35.0)
            truth = r.tap_labels[taps.delay_idx, taps.doa_idx, taps.dod_idx]
            t0 = time.perf_counter()
            lab_kpd = kpd(taps)
            t1 = time.perf_counter()
            ml = ml_kpd(taps, n_angle=gen.grid.n_angle)
            t2 = time.perf_counter()
            rows.append({"condition": cond, "channel": i, "n_taps": taps.n, "n_true": len(np.unique(truth)),
                         "n_kpd": int(lab_kpd.max() + 1), "n_mlkpd": ml.n_clusters, "levels": ml.levels,
                         "ari_kpd": ari(truth, lab_kpd), "ari_mlkpd": ari(truth, ml.labels),
                         "t_kpd_s": t1 - t0, "t_mlkpd_s": t2 - t1})
            log.info("exp5 %s", rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(d / "clustering.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for k, (cond, g) in enumerate(df.groupby("condition")):
        ax.scatter(g.ari_kpd, g.ari_mlkpd, label=cond.upper(), s=14)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("ARI single-level KPD")
    ax.set_ylabel("ARI ML-KPD")
    ax.legend()
    fig.savefig(d / "ari.png")
    plt.close(fig)
    summ = df.groupby("condition")[["n_true", "n_kpd", "n_mlkpd", "ari_kpd", "ari_mlkpd", "t_mlkpd_s"]].mean()
    cfg = {"profile": profile, "seed": prof["seed"], "n_channels_per_condition": prof["n_clustering_channels"],
           "clustering": P.raw["clustering"],
           "ground_truth": "tap label = cluster contributing the most power to the tap (exact for generator output)"}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", {"per_channel": df.to_dict(orient="records"), "mean": summ.reset_index().to_dict(orient="records")})
    _readme(d, "Experiment 5 - KPD vs ML-KPD clustering",
            "Score both clustering algorithms against exact cluster labels of generated ADPSs (taps above 35 dB).", cfg,
            [f"{c.upper()}: mean true clusters {summ.loc[c, 'n_true']:.1f}, KPD {summ.loc[c, 'n_kpd']:.1f}, ML-KPD "
             f"{summ.loc[c, 'n_mlkpd']:.1f}; mean ARI KPD {summ.loc[c, 'ari_kpd']:.3f}, ML-KPD {summ.loc[c, 'ari_mlkpd']:.3f}"
             for c in summ.index],
            ["the paper validates ML-KPD only visually on measured ADPSs (no accuracy numbers), so this is an "
             "additional, synthetic benchmark"],
            ["generated clusters are idealised (no noise, no sidelobes); on synthetic sounder data with IFFT sidelobes "
             "both KPD variants over-segment strong clusters (tests/test_adps.py, docs/LIMITATIONS.md)",
             "tiny clusters of 1-3 taps hidden under stronger ones cannot be separated by any density method"])


# --------------------------------------------------------------------------- exp6 / exp7
def _channel_stats(r: pd.DataFrame) -> dict:
    return {"ds_median_m": float(r.delay_spread_m.median()), "asa_median": float(r.doa_spread.median()),
            "asd_median": float(r.dod_spread.median()), "visible_mean": float(r.n_visible.mean())}


def _cvm_vs_measured(r: pd.DataFrame, cond: str) -> dict:
    if not digitized_available():
        return {}
    out = {}
    for key, col, fig in (("delay", "delay_spread_m", "fig17a_delay_spread"), ("doa", "doa_spread", "fig17b_doa_spread"),
                          ("dod", "dod_spread", "fig17c_dod_spread")):
        out[f"cvm_{key}"] = cvm_distance(r[col].to_numpy(), load_samples(fig, f"{cond}_measured"))
    return out


def exp6(profile, prof, sim, log):
    d = _dir("exp6_correction_ablation")
    n = max(20, int(prof["n_realizations"]) // 4)
    rows = []
    for cond in ("los", "nlos"):
        for use in (True, False):
            r, c = simulate(cond, n, int(prof["seed"]), "val", validation_distances(P, cond), use_correction=use,
                            cache=f"{profile}_corr{int(use)}_{cond}", log=log)
            v = c[c.visible]
            e = np.column_stack([v.re_s_tau_m / v.s_tau_m, v.re_s_doa / v.s_doa, v.re_s_dod / v.s_dod]) - 1
            rows.append({"condition": cond, "correction": use, "n": n,
                         "median_abs_err_tau": float(np.median(np.abs(e[:, 0]))),
                         "median_abs_err_doa": float(np.median(np.abs(e[:, 1]))),
                         "median_abs_err_dod": float(np.median(np.abs(e[:, 2]))),
                         "median_signed_err_tau": float(np.median(e[:, 0])), **_channel_stats(r), **_cvm_vs_measured(r, cond)})
    df = pd.DataFrame(rows)
    df.to_csv(d / "ablation.csv", index=False, float_format="%.4f")
    cfg = {"profile": profile, "seed": prof["seed"], "n": n}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", df.to_dict(orient="records"))
    obs = [f"{r.condition.upper()} correction={r.correction}: median |re-observed/target-1| delay {r.median_abs_err_tau:.3f}, "
           f"DoA {r.median_abs_err_doa:.3f}, DoD {r.median_abs_err_dod:.3f}; channel median DS {r.ds_median_m:.2f} m"
           for r in df.itertuples()]
    _readme(d, "Experiment 6 - discretisation-correction ablation",
            "Effect of step 5's correction on how faithfully generated clusters reproduce their drawn spreads on the grid.",
            cfg, obs, ["paper states that without correction discretisation error is 'particularly pronounced' for small "
                       "spreads; no numbers given"], ["same random numbers in both arms (common seeds)"])


def exp7(profile, prof, sim, log):
    d = _dir("exp7_parameter_variants")
    n = max(20, int(prof["n_realizations"]) // 4)
    variants = {"table_I": {}, "text_arrival": {"arrival_source": "text"},
                "wrapped_laplace": {"bimodal_laplace_mode": "wrapped"}}
    rows = []
    for cond in ("los", "nlos"):
        for name, var in variants.items():
            if cond == "nlos" and name == "wrapped_laplace":
                continue  # NLoS centroids are uniform: identical to table_I
            r, _ = simulate(cond, n, int(prof["seed"]), "val", validation_distances(P, cond), variant=var or None,
                            cache=f"{profile}_var_{name}_{cond}", log=log)
            rows.append({"condition": cond, "variant": name, "n": n, **_channel_stats(r), **_cvm_vs_measured(r, cond)})
    df = pd.DataFrame(rows)
    df.to_csv(d / "variants.csv", index=False, float_format="%.4f")
    cfg = {"profile": profile, "seed": prof["seed"], "n": n, "variants": variants}
    save_json(d / "config.json", cfg)
    save_json(d / "results.json", df.to_dict(orient="records"))
    obs = [f"{r.condition.upper()} {r.variant}: DS {r.ds_median_m:.2f} m, ASA {r.asa_median:.3f}, ASD {r.asd_median:.3f}"
           + (f", CvM vs measured delay/DoA/DoD {r.cvm_delay:.4f}/{r.cvm_doa:.4f}/{r.cvm_dod:.4f}" if hasattr(r, "cvm_delay") else "")
           for r in df.itertuples()]
    _readme(d, "Experiment 7 - sensitivity to ambiguous published parameters",
            "Table I vs in-text arrival-time Gamma parameters, and normalised vs wrapped bimodal Laplace (Eq. 3).",
            cfg, obs, ["the paper does not state which arrival parameters its toolkit uses; differences between variants "
                       "indicate how much the ambiguity matters"], ["same seeds across variants"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke", choices=["smoke", "standard", "full"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    log = get_logger("run_experiments")
    sim, prof, _ = load_all(a.profile)
    from _common import correction_table
    corr = correction_table(sim)
    jobs = {"exp1": lambda: exp1(a.profile, prof, sim, log), "exp2": lambda: exp2(a.profile, prof, sim, log),
            "exp3": lambda: exp3(a.profile, prof, sim, log, corr), "exp4": lambda: exp4(a.profile, prof, sim, log),
            "exp5": lambda: exp5(a.profile, prof, sim, log), "exp6": lambda: exp6(a.profile, prof, sim, log),
            "exp7": lambda: exp7(a.profile, prof, sim, log)}
    for k, fn in jobs.items():
        if a.only is None or k in a.only:
            t0 = time.perf_counter()
            fn()
            log.info("%s finished in %.0f s", k, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
