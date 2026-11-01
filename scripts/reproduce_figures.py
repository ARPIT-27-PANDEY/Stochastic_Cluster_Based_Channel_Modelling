"""Reproduce Figs. 2-17 of the paper (and supporting figures) at 300 dpi.

    python scripts/reproduce_figures.py --profile smoke|standard|full [--only fig14 fig15]

Every curve is labelled by its origin:
  * "measured (digitized)"        - step CDF read from the paper's vector figure (not raw data)
  * "paper synthetic (digitized)" - the authors' generator output read from Fig. 17
  * "Table I model"               - analytic distribution with the published parameters
  * "ours: drawn" / "ours: observable" - OUR SIMULATION with the published parameters
Run generate_channels.py first; Figs. 12-13 also use the MLE of run_experiments.py (exp4)
when available.  A catalogue is written to results/figures/FIGURES.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import (COLORS, EXP_DIR, FIG_DIR, ROOT, get_logger, load_all, make_generator, setup_mpl, simulate,
                     validation_distances)
from channel_model import distributions as D
from channel_model import load_channel_parameters
from channel_model.dynamic_range import (AngleGrid, continuous_observables, discrete_observables,
                                         infinite_dr_angular_spread, regime_boundaries_db)
from channel_model.inverse_mapping import invert_spreads
from channel_model.parameters import LOS, LOS_CLUSTER, NLOS
from channel_model.power_model import ncp_mean, ncp_std
from channel_model.validation import digitized_available, load_curve

plt = setup_mpl()
P = load_channel_parameters()
CATALOGUE: list[dict] = []


def note(fig: str, title: str, content: str, data: str, cls: str, remarks: str = "") -> None:
    CATALOGUE.append({"figure": fig, "title": title, "content": content, "data": data, "classification": cls,
                      "remarks": remarks})


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def _dig(figname: str, role: str):
    if not digitized_available():
        return None
    try:
        return load_curve(figname, role)
    except OSError:
        return None


def _ecdf_line(ax, x, **kw):
    x = np.sort(np.asarray(x)[np.isfinite(x)])
    if x.size:
        ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", **kw)


# --------------------------------------------------------------------------- data
def datasets(profile: str, log):
    _, prof, _ = load_all(profile)
    out = {}
    for cond in ("los", "nlos"):
        out[cond] = simulate(cond, int(prof["n_realizations"]), int(prof["seed"]), "val",
                             validation_distances(P, cond), cache=f"{profile}_val_{cond}", log=log)
    return out


# --------------------------------------------------------------------------- Fig. 2
def fig02(ds):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.logspace(0, np.log10(300), 400)
    for cond, c in (("los", COLORS["los"]), ("nlos", COLORS["nlos"])):
        d = _dig("fig02_cluster_number", f"{cond}_measured")
        if d is not None:
            ax.step(*d, where="post", color=c, lw=2.2, alpha=0.35, label=f"{cond.upper()} measured (digitized)")
        real = ds[cond][0]
        _ecdf_line(ax, real.n_clusters, color=c, lw=1, ls="--", label=f"{cond.upper()} ours: drawn N")
        _ecdf_line(ax, real.n_visible, color=c, lw=1.2, label=f"{cond.upper()} ours: observable (35 dB)")
    ax.plot(x, D.gamma_cdf(x - 1, P.n_los_additional.shape, P.n_los_additional.scale), "k:", lw=1,
            label="Table I Gamma (N_LoS-1 = additional)")
    ax.plot(x, D.gamma_cdf(x - 1, P.n_nlos_minus_one.shape, P.n_nlos_minus_one.scale), "k-.", lw=1,
            label="Table I Gamma (N_NLoS-1)")
    ax.set_xscale("log")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("CDF")
    ax.set_xlim(1, 300)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    ax.set_title("Fig. 2 - number of clusters")
    save(fig, "fig02_cluster_number")
    note("fig02", "Number of clusters (CDF)", "measured (digitized) vs Table I Gamma vs our drawn / observable counts",
         "digitized + simulated", "Statistical reproduction using published parameters",
         f"ours observable mean: LoS {ds['los'][0].n_visible.mean():.1f}, NLoS {ds['nlos'][0].n_visible.mean():.1f}; "
         f"drawn mean LoS {ds['los'][0].n_clusters.mean():.1f}, NLoS {ds['nlos'][0].n_clusters.mean():.1f}")


# --------------------------------------------------------------------------- Figs. 3-5
def _arrival(ds, cond, cls, figname, title, num):
    cl = ds[cond][1]
    sel = cl[cl.cls == cls]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bins = np.arange(0, 310, 10)
    ax.hist(sel.arrival_m, bins=bins, density=True, color="0.75", label="ours: drawn (all clusters)")
    ax.hist(sel.arrival_m[sel.visible], bins=bins, density=True, histtype="step", color=COLORS[cond], lw=1.4,
            label="ours: observable clusters")
    x = np.linspace(0.1, 300, 600)
    g = P.arrival[cond]
    alt = P.raw["arrival_time_m"]["text"][cond]
    ax.plot(x, D.gamma_pdf(x, g.shape, g.scale), "k-", lw=1.2, label=f"Table I Gamma({g.shape}, {g.scale})")
    ax.plot(x, D.gamma_pdf(x, alt["shape"], alt["scale"]), "k--", lw=0.9,
            label=f"text Gamma({alt['shape']}, {alt['scale']})")
    ax.set_xlabel("cluster arrival time (excess delay) [m]")
    ax.set_ylabel("PDF")
    ax.set_xlim(0, 300)
    ax.legend()
    ax.set_title(f"Fig. {num} - {title}")
    save(fig, figname)
    note(f"fig{num:02d}", title, "histogram of our simulated arrival times (drawn / observable) with Table I and text Gamma",
         "simulated + analytic", "Statistical reproduction using published parameters",
         "the paper's measured histogram is not digitized (bars); the observable histogram shows the DR effect: "
         f"mean drawn {sel.arrival_m.mean():.1f} m vs observable {sel.arrival_m[sel.visible].mean():.1f} m")


def fig03_04_05(ds):
    _arrival(ds, "los", LOS, "fig03_arrival_los", "cluster arrival time, LoS", 3)
    _arrival(ds, "nlos", NLOS, "fig04_arrival_nlos", "cluster arrival time, NLoS", 4)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bins = np.arange(0, 310, 10)
    means = {}
    for cond in ("los", "nlos"):
        cl = ds[cond][1]
        sel = cl[cl.visible_ld15]
        means[cond] = ds[cond][0].n_visible_ld15.mean()
        ax.hist(sel.arrival_m, bins=bins, density=True, histtype="step", lw=1.4, color=COLORS[cond],
                label=f"{cond.upper()} ours (LD: no LoS cluster, 15 dB); mean N = {means[cond]:.1f}")
    ax.set_xlabel("cluster arrival time [m]")
    ax.set_ylabel("PDF")
    ax.set_xlim(0, 300)
    ax.legend()
    ax.set_title("Fig. 5 - arrival time under the LD setting (simulated)")
    save(fig, "fig05_arrival_ld_setting")
    note("fig05", "Arrival time under the LD setting", "our simulated channels with the LoS cluster removed and a 15 dB DR",
         "simulated", "Approximate reproduction",
         f"mean LD cluster number ours LoS {means['los']:.1f} / NLoS {means['nlos']:.1f} vs published 3 / 9 "
         "(the paper's values are measured; ours come from the generator)")


# --------------------------------------------------------------------------- Figs. 6, 10, 11
def _spread_cdf(ds, figname, digname, key_model, col_re, xlabel, num, xlim):
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    x = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
    devs = {}
    for cls, cond, label, color in ((LOS_CLUSTER, "los", "LoS cluster", COLORS["loscluster"]),
                                    (LOS, "los", "LoS", COLORS["los"]), (NLOS, "nlos", "NLoS", COLORS["nlos"])):
        role = {LOS_CLUSTER: "loscluster", LOS: "los", NLOS: "nlos"}[cls]
        d = _dig(digname, f"{role}_measured")
        if d is not None:
            ax.step(*d, where="post", color=color, lw=2.2, alpha=0.3, label=f"{label} measured (digitized)")
        m = getattr(P, key_model)[{LOS_CLUSTER: "los_cluster", LOS: "los", NLOS: "nlos"}[cls]]
        ax.plot(x, D.lognormal_cdf(x, m.mu, m.sigma), color=color, lw=0.9, ls=":", label=f"{label} Table I")
        cl = ds[cond][1]
        sel = cl[(cl.cls == cls) & cl.visible]
        _ecdf_line(ax, sel[col_re], color=color, lw=1.1, label=f"{label} ours: re-observed")
        if d is not None and len(sel):
            from channel_model.validation import cvm_distance, load_samples  # noqa
            devs[label] = float(np.max(np.abs(d[1] - D.lognormal_cdf(d[0], m.mu, m.sigma))))
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.legend(fontsize=5.6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title(f"Fig. {num}")
    save(fig, figname)
    return devs


def fig06_10_11(ds):
    d6 = _spread_cdf(ds, "fig06_intra_cluster_delay_spread", "fig06_delay_spread", "delay_spread", "re_s_tau_m",
                     "intra-cluster delay spread [m]", 6, (0.05, 5))
    d10 = _spread_cdf(ds, "fig10_intra_cluster_doa_spread", "fig10_doa_spread", "doa_spread", "re_s_doa",
                      "intra-cluster DoA angular spread", 10, (0.05, 0.6))
    d11 = _spread_cdf(ds, "fig11_intra_cluster_dod_spread", "fig11_dod_spread", "dod_spread", "re_s_dod",
                      "intra-cluster DoD angular spread", 11, (0.05, 0.6))
    for f, t, d in (("fig06", "Intra-cluster delay spread", d6), ("fig10", "Intra-cluster DoA spread", d10),
                    ("fig11", "Intra-cluster DoD spread", d11)):
        note(f, t, "measured (digitized) vs Table I log-normal vs spreads re-observed on the 1 ns / 10 deg grid "
             "from our generated clusters", "digitized + simulated", "Statistical reproduction using published parameters",
             "max |digitized measured CDF - Table I CDF|: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()))


# --------------------------------------------------------------------------- Figs. 7-9
def _centroid(ds, cond, col, model_key, figname, title, num):
    cl = ds[cond][1]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bins = np.linspace(-np.pi, np.pi, 37)
    sel = cl[cl.cls != LOS_CLUSTER]
    ax.hist(sel[col], bins=bins, density=True, color="0.75", label="ours: drawn")
    ax.hist(sel[col][sel.visible], bins=bins, density=True, histtype="step", color=COLORS[cond], lw=1.4,
            label="ours: observable")
    x = np.linspace(-np.pi, np.pi, 1441)
    m = getattr(P, model_key)[cond]
    if m.type == "uniform":
        ax.plot(x, np.full_like(x, 1 / (2 * np.pi)), "k-", label="Table I uniform")
    else:
        ax.plot(x, D.bimodal_laplace_pdf(x, m.weights, m.centers, m.decays, "truncated"), "k-", lw=1,
                label="Table I bimodal Laplace (Eq. 3)")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 5)
    ax.set_xlabel(f"cluster {col.upper()} centroid [rad]")
    ax.set_ylabel("PDF")
    ax.legend()
    ax.set_title(f"Fig. {num} - {title}")
    save(fig, figname)


def fig07_08_09(ds):
    _centroid(ds, "los", "doa", "doa_centroid", "fig07_doa_centroid_los", "DoA centroids, LoS", 7)
    _centroid(ds, "los", "dod", "dod_centroid", "fig08_dod_centroid_los", "DoD centroids, LoS", 8)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    cl = ds["nlos"][1]
    bins = np.linspace(-np.pi, np.pi, 37)
    for ax, col in zip(axs, ("doa", "dod")):
        ax.hist(cl[col], bins=bins, density=True, color="0.75", label="ours: drawn")
        ax.hist(cl[col][cl.visible], bins=bins, density=True, histtype="step", color=COLORS["nlos"], lw=1.4,
                label="ours: observable")
        ax.axhline(1 / (2 * np.pi), color="k", label="Table I uniform")
        ax.set_xlabel(f"{col.upper()} centroid [rad]")
    axs[0].set_ylabel("PDF")
    axs[0].legend()
    fig.suptitle("Fig. 9 - DoA/DoD centroids, NLoS")
    save(fig, "fig09_centroids_nlos")
    for f, t in (("fig07", "DoA centroids, LoS"), ("fig08", "DoD centroids, LoS"), ("fig09", "DoA/DoD centroids, NLoS")):
        note(f, t, "histogram of our simulated centroids (drawn / observable) with the Table I density",
             "simulated + analytic", "Statistical reproduction using published parameters",
             "Eq. (3) normalised on [-pi, pi) ('truncated' mode); measured histogram not digitized")


# --------------------------------------------------------------------------- Figs. 12-13
def fig12_13(ds):
    exp4 = EXP_DIR / "exp4_power_mle" / "results.json"
    fits = json.loads(exp4.read_text()) if exp4.exists() else None
    for cond, num in (("los", 12), ("nlos", 13)):
        cl = ds[cond][1]
        cls = LOS if cond == "los" else NLOS
        sel = cl[(cl.cls == cls) & cl.visible]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(sel), size=min(4000, len(sel)), replace=False)
        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        x = np.linspace(0, 20, 100)
        pp = P.power[cond]
        mu, sg = ncp_mean(x, pp), ncp_std(x, pp)
        ax.fill_between(x, mu - 3 * sg, mu + 3 * sg, color="0.85", label="Table I +-3 sigma")
        ax.fill_between(x, mu - sg, mu + sg, color="0.7", label="Table I +-1 sigma")
        ax.plot(x, mu, "k-", lw=1.4, label="Table I mean (ground truth)")
        ax.scatter(sel.ratio_db.iloc[idx], sel.ncp_db.iloc[idx], s=4, color=COLORS[cond], alpha=0.5, zorder=3,
                   label="ours: observable clusters (simulated)")
        if fits and cond in fits:
            f = fits[cond]
            ax.plot(x, f["mle"]["alpha"] + f["mle"]["beta"] * x, color="#9467bd", lw=1.2, ls="--",
                    label="ours: truncated MLE")
            ax.plot(x, f["ols"]["alpha"] + f["ols"]["beta"] * x, color="#e377c2", lw=1.2, ls="-.",
                    label="ours: OLS (no truncation)")
        ax.set_xlabel("distance ratio 10log10(d/d_LoS) [dB]")
        ax.set_ylabel("normalised cluster power [dB]")
        ax.set_xlim(0, 20)
        ax.legend(fontsize=6)
        ax.set_title(f"Fig. {num} - NCP vs distance ratio, {cond.upper()}")
        save(fig, f"fig{num}_ncp_{cond}")
        note(f"fig{num}", f"NCP vs distance ratio ({cond.upper()})",
             "observable simulated clusters, Table I mean/sigma bands, our truncated MLE and OLS fits (exp4)",
             "simulated", "Independent numerical reproduction (method) / statistical (data)",
             "data are synthetic with KNOWN ground truth (Table I), so the plot tests the estimator, "
             "not the measurement fit")


# --------------------------------------------------------------------------- Figs. 14-15
def fig14_15(sim):
    c = sim["fig14_15"]
    kt, ka, kd = c["kappa_tau_m"], c["kappa_doa_rad"], c["kappa_dod_rad"]
    dr = np.arange(c["dr_db_min"], c["dr_db_max"] + 0.5 * c["dr_db_step"], c["dr_db_step"], dtype=float)
    fine_res = c["fine_delay_resolution_m"]
    fine = discrete_observables(kt / fine_res, ka, kd, dr, AngleGrid(c["fine_angle_step_deg"]), fine_res)
    native = discrete_observables(kt / 0.29979, ka, kd, dr, AngleGrid(10.0), 0.29979)
    s_act = np.column_stack([fine["s_tau"], fine["s_r"], fine["s_t"]])
    inv = invert_spreads(s_act, dr)
    re = continuous_observables(inv.kappa[:, 0], inv.kappa[:, 1], inv.kappa[:, 2], dr)
    b = regime_boundaries_db(ka, kd)
    shades = ["#fde0dd", "#e0ecf4", "#e5f5e0", "#fff7bc"]

    def regimes(ax):
        edges = [dr[0], *b, dr[-1]]
        for i in range(4):
            ax.axvspan(edges[i], edges[i + 1], color=shades[i], alpha=0.7, lw=0)
        for i, txt in enumerate(["(i)", "(ii)", "(iii)", "(iv)"]):
            ax.text(0.5 * (edges[i] + edges[i + 1]), 0.02, txt, transform=ax.get_xaxis_transform(), ha="center", fontsize=7)

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    regimes(ax)
    ax.plot(dr, fine["mean_tau"], "-", color="#1f77b4", lw=1.4, label="mean delay, actual (fine grid)")
    ax.plot(dr, re["mean_tau"], "o", ms=2.5, mfc="none", color="#1f77b4", label="mean delay, inverse-solved")
    ax.plot(dr, fine["s_tau"], "-", color="#d62728", lw=1.4, label="delay spread, actual (fine grid)")
    ax.plot(dr, re["s_tau"], "s", ms=2.5, mfc="none", color="#d62728", label="delay spread, inverse-solved")
    ax.plot(dr, native["s_tau"], ":", color="#d62728", lw=1, label="delay spread on 1 ns / 10 deg grid")
    ax.plot(dr[::4], np.full(dr[::4].size, kt), "kd--", ms=3, lw=0.8, label="ideal (kappa_tau = 10 m)")
    ax.set_xlabel("dynamic range [dB]")
    ax.set_ylabel("[m]")
    ax.set_xlim(dr[0], dr[-1])
    ax.legend(fontsize=5.8, loc="center right")
    ax.set_title("Fig. 14 - inverse estimation, delay statistics")
    save(fig, "fig14_inverse_delay")

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    regimes(ax)
    for j, name, kap, col in ((1, "DoA", ka, "#2ca02c"), (2, "DoD", kd, "#9467bd")):
        key = "s_r" if j == 1 else "s_t"
        ax.plot(dr, fine[key], "-", color=col, lw=1.4, label=f"{name} spread, actual (kappa = {kap / np.pi:.1f} pi)")
        ax.plot(dr, re[key], "o", ms=2.5, mfc="none", color=col, label=f"{name} spread, inverse-solved")
        ax.plot(dr, native[key], ":", color=col, lw=1, label=f"{name} spread on 10 deg grid")
        ax.plot(dr[::4], np.full(dr[::4].size, infinite_dr_angular_spread(kap)), "d--", color=col, ms=3, lw=0.8,
                label=f"{name} DR -> inf (truncated Laplace)")
        ax.axhline(np.sqrt(1 - 1 / (1 + kap**2) ** 2), color=col, lw=0.6, ls="-.")
    ax.set_xlabel("dynamic range [dB]")
    ax.set_ylabel("Fleury angular spread")
    ax.set_xlim(dr[0], dr[-1])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5.4, loc="lower right")
    ax.set_title("Fig. 15 - inverse estimation, angular statistics")
    save(fig, "fig15_inverse_angular")
    err = np.max(np.abs(np.column_stack([re["s_tau"], re["s_r"], re["s_t"]]) - s_act) / s_act, axis=0)
    pd.DataFrame({"dr_db": dr, "mean_tau_actual": fine["mean_tau"], "s_tau_actual": fine["s_tau"],
                  "s_doa_actual": fine["s_r"], "s_dod_actual": fine["s_t"], "kappa_tau_est": inv.kappa[:, 0],
                  "kappa_doa_est": inv.kappa[:, 1], "kappa_dod_est": inv.kappa[:, 2],
                  "s_tau_reobserved": re["s_tau"], "s_doa_reobserved": re["s_r"], "s_dod_reobserved": re["s_t"],
                  "s_tau_native_grid": native["s_tau"], "s_doa_native_grid": native["s_r"],
                  "s_dod_native_grid": native["s_t"]}).to_csv(ROOT / "results" / "tables" / "fig14_15_data.csv",
                                                              index=False, float_format="%.6g")
    for f, t in (("fig14", "Inverse estimation - delay statistics"), ("fig15", "Inverse estimation - angular statistics")):
        note(f, t, "actual spreads of a truncated exponential-Laplacian cluster (fine grid 0.01 m / 0.25 deg) vs "
             "spreads re-observed after inverse mapping; native-grid curves show the discretisation error",
             "simulated (deterministic)", "Independent numerical reproduction",
             f"regime boundaries {b[0]:.2f} / {b[1]:.2f} / {b[2]:.2f} dB; max relative re-observation error "
             f"(fine grid, without correction) delay {err[0]:.3%}, DoA {err[1]:.3%}, DoD {err[2]:.3%}. "
             "kappa_DoD = 0.2 pi, kappa_DoA = 0.3 pi as in the text (legend of the paper's figure suggests the reverse)")


# --------------------------------------------------------------------------- Fig. 16
def fig16():
    steps = [("1. Inputs: d_LoS, condition (LoS/NLoS), DR = 35 dB", 1),
             ("2. Number of clusters: LoS cluster + Gamma(3.39, 3.35)\nor 1 + Gamma(1.89, 46.72) (NLoS)", 2),
             ("3. Per cluster: arrival (Gamma), DoA/DoD centroid\n(bimodal Laplace / uniform), spreads (log-normal),\n"
              "NCP ~ N(alpha + beta x, (rho + eta x)^2)", 3),
             ("4. Global cut-off = strongest cluster peak - DR", 1),
             ("5. Discretisation correction + inverse mapping\nS_obs -> kappa = F^-1(S_obs; DR_eff)  (Newton)", 2),
             ("6. Cluster ADPS = exp-Laplacian(kappa) scaled to its\npower, placed at its centroid (Eq. 12)", 2),
             ("7. Superposition + global 35 dB threshold -> ADPS", 1)]
    line_h, gap, pad = 0.034, 0.045, 0.022
    heights = [n * line_h + 2 * pad for _, n in steps]
    total = sum(heights) + gap * (len(steps) - 1)
    fig, ax = plt.subplots(figsize=(4.8, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    from matplotlib.patches import FancyBboxPatch
    top = 0.5 + total / 2
    centers = []
    for (txt, n), h in zip(steps, heights):
        y0 = top - h
        ax.add_patch(FancyBboxPatch((0.04, y0), 0.92, h, boxstyle="round,pad=0.005,rounding_size=0.015",
                                    fc="#eef3fb", ec="#1f4e79", lw=0.9))
        ax.text(0.5, y0 + h / 2, txt, ha="center", va="center", fontsize=7.3, family="monospace")
        centers.append((y0, top))
        top = y0 - gap
    for (y0, _), (_, nt) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(0.5, nt), xytext=(0.5, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#1f4e79"))
    # fixed-point loop 6 -> 4
    y4 = (centers[3][0] + centers[3][1]) / 2
    y6 = (centers[5][0] + centers[5][1]) / 2
    ax.annotate("", xy=(0.962, y4), xytext=(0.962, y6),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4", connectionstyle="bar,fraction=0.25"))
    ax.text(1.08, (y4 + y6) / 2, "peaks depend\non shapes:\niterate 4-6", fontsize=6, color="0.35", va="center",
            ha="left", style="italic")
    ax.set_title("Fig. 16 - synthetic channel generation (as implemented)")
    save(fig, "fig16_generation_flow")
    note("fig16", "Generation flow diagram", "flow of our implementation of the 7 steps", "diagram",
         "Exact reproduction (structure)", "adds the explicit peak/cut-off fixed point that the paper leaves implicit")


# --------------------------------------------------------------------------- Fig. 17
def fig17(ds):
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.9))
    for ax, (col, digname, xlabel, xlim) in zip(axs, (("delay_spread_m", "fig17a_delay_spread", "delay spread [m]", (0, 40)),
                                                      ("doa_spread", "fig17b_doa_spread", "DoA angular spread", (0, 1)),
                                                      ("dod_spread", "fig17c_dod_spread", "DoD angular spread", (0, 1)))):
        for cond in ("los", "nlos"):
            c = COLORS[cond]
            d = _dig(digname, f"{cond}_measured")
            if d is not None:
                ax.step(*d, where="post", color=c, lw=2.4, alpha=0.3, label=f"{cond.upper()} measured (digitized)")
            d = _dig(digname, f"{cond}_paper_synthetic")
            if d is not None:
                ax.step(*d, where="post", color=c, lw=1, ls="--", label=f"{cond.upper()} paper synthetic (digitized)")
            _ecdf_line(ax, ds[cond][0][col], color=c, lw=1.3, label=f"{cond.upper()} ours (simulated)")
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
    axs[0].set_ylabel("CDF")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig. 17 - channel statistics: measured vs paper synthetic vs ours")
    save(fig, "fig17_validation_cdfs")
    note("fig17", "Validation CDFs (delay / DoA / DoD spread)",
         "digitized measured and paper-synthetic CDFs vs our generator (channels at the measured link distances)",
         "digitized + simulated", "Statistical reproduction using published parameters",
         "CvM distances in results/tables/cvm_validation.csv")


# --------------------------------------------------------------------------- supporting
def fig_examples(sim):
    gen = make_generator(sim)
    fig, axs = plt.subplots(2, 3, figsize=(9.2, 5.2))
    for row, (cond, d, seed) in enumerate((("los", 20.0, 1), ("nlos", 40.0, 3))):
        r = gen.generate(d, cond, np.random.default_rng(seed), keep_adps=True)
        A = r.adps
        with np.errstate(divide="ignore"):
            dd = 10 * np.log10(A.max(axis=2) / A.max())
            dt = 10 * np.log10(A.max(axis=1) / A.max())
            pdp = 10 * np.log10(A.max(axis=(1, 2)) / A.max())
        tau = gen.grid.tau_m
        deg = np.rad2deg(gen.grid.phi)
        kmax = max(10, int(np.nonzero(A.max(axis=(1, 2)) > 0)[0].max()) + 5)
        ext = [deg[0] - 5, deg[-1] + 5, tau[kmax], tau[0]]
        for ax, img, ttl in ((axs[row, 0], dd[:kmax], "delay-DoA"), (axs[row, 1], dt[:kmax], "delay-DoD")):
            im = ax.imshow(np.where(np.isfinite(img), img, np.nan), aspect="auto", extent=ext, vmin=-35, vmax=0,
                           cmap="viridis", interpolation="nearest")
            ax.set_xlabel(f"{ttl.split('-')[1]} [deg]")
            ax.set_ylabel("excess delay [m]")
            ax.set_title(f"{cond.upper()} d={d:g} m: {ttl} (max over other angle)", fontsize=7)
        fig.colorbar(im, ax=axs[row, 1], label="dB")
        axs[row, 2].plot(tau[:kmax], pdp[:kmax], lw=0.8)
        axs[row, 2].set_ylim(-36, 1)
        axs[row, 2].set_xlabel("excess delay [m]")
        axs[row, 2].set_ylabel("omni PDP [dB]")
        axs[row, 2].set_title(f"{r.n_visible}/{r.clusters.n} clusters observable, {r.metrics['n_taps']} taps", fontsize=7)
    fig.suptitle("Example synthetic ADPS realizations (simulated, 35 dB)")
    fig.tight_layout()
    save(fig, "figS1_example_adps")
    note("figS1", "Example synthetic ADPS", "one LoS and one NLoS realization from our generator", "simulated",
         "Independent numerical reproduction", "supporting figure (not in the paper)")


def fig_pathloss(profile, log):
    _, prof, _ = load_all(profile)
    from _common import uniform_distances
    from channel_model.path_loss import fit_alpha_beta
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, cond in zip(axs, ("los", "nlos")):
        n = int(prof["n_realizations"])
        real, _ = simulate(cond, n, int(prof["seed"]), "pl", uniform_distances(n, int(prof["seed"]), cond),
                           cache=f"{profile}_pl_{cond}", log=log)
        d = real.d_m.to_numpy()
        for col, mk, lab in (("pg_omni_db", "o", "omni"), ("pg_maxdir_db", "x", "max-dir")):
            pl = -real[col].to_numpy()
            f = fit_alpha_beta(d, pl)
            ax.scatter(d, pl, s=4, marker=mk, alpha=0.4, label=f"ours {lab} (simulated)")
            x = np.logspace(0, 2, 50)
            ax.plot(x, f.alpha + 10 * f.beta * np.log10(x), lw=1,
                    label=f"{lab}: alpha={f.alpha:.1f}, beta={f.beta:.2f}, sigma={f.sigma:.1f}")
        pub = __import__("yaml").safe_load((ROOT / "config" / "channel_parameters.yaml").read_text())
        ref = pub["published_benchmarks"]["path_loss_table_ii"]["alpha_beta"][f"{cond}_omni"]
        x = np.logspace(0, 2, 50)
        ax.plot(x, ref["alpha_ref"] + 10 * ref["beta_ref"] * np.log10(x), "k--", lw=1, label="[21] measured omni fit")
        ax.set_xscale("log")
        ax.set_xlabel("Tx-Rx distance [m]")
        ax.set_title(cond.upper())
        ax.legend(fontsize=5.5)
    axs[0].set_ylabel("path loss [dB]")
    fig.suptitle("Path loss of generated channels (all pool realizations)")
    save(fig, "figS2_pathloss")
    note("figS2", "Path loss of generated channels", "PL = -10log10(sum omni / max-dir PDP) vs distance with alpha-beta fit",
         "simulated", "Statistical reproduction using published parameters", "Table II comparison in results/tables/pathloss_table2.csv")


def write_catalogue(profile):
    lines = ["# Figure catalogue", "", f"Generated by `scripts/reproduce_figures.py --profile {profile}`. "
             "All PNGs are 300 dpi. 'digitized' = read from the paper's vector figures (not raw measurement data); "
             "'simulated' = output of this repository's generator using the published parameters.", ""]
    for c in sorted(CATALOGUE, key=lambda x: x["figure"]):
        lines += [f"## {c['figure']} - {c['title']}", f"- **Content:** {c['content']}", f"- **Data:** {c['data']}",
                  f"- **Classification:** {c['classification']}"]
        if c["remarks"]:
            lines.append(f"- **Remarks:** {c['remarks']}")
        lines.append("")
    (FIG_DIR / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke", choices=["smoke", "standard", "full"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    log = get_logger("reproduce_figures")
    sim, _, _ = load_all(a.profile)
    want = lambda k: a.only is None or k in a.only  # noqa: E731
    ds = datasets(a.profile, log) if any(want(k) for k in ("fig02", "fig03", "fig06", "fig07", "fig12", "fig17")) else None
    jobs = [("fig02", lambda: fig02(ds)), ("fig03", lambda: fig03_04_05(ds)), ("fig06", lambda: fig06_10_11(ds)),
            ("fig07", lambda: fig07_08_09(ds)), ("fig12", lambda: fig12_13(ds)), ("fig14", lambda: fig14_15(sim)),
            ("fig16", fig16), ("fig17", lambda: fig17(ds)), ("figS1", lambda: fig_examples(sim)),
            ("figS2", lambda: fig_pathloss(a.profile, log))]
    for key, fn in jobs:
        if want(key):
            fn()
            log.info("%s done", key)
    if a.only is None:
        write_catalogue(a.profile)


if __name__ == "__main__":
    main()
