"""Run the GNU Radio sounder flowgraph and compare its output with the paper.

    python scripts/run_gnuradio.py --run --analyse [--sets table1 calibrated] [--reps 10] [--n-pl 300]

--run      compiles gnuradio/subthz_sounder_batch.grc with grcc and runs the generated
           flowgraph (GNU Radio's Python, radioconda) in parallel processes:
             * 'val' data sets: channels at the 21 LoS / 17 NLoS measured link distances (Fig. 17)
             * 'pl'  data sets: channels at d ~ U(1, 100) m (Table II path loss)
           for each parameter set: 'table1' (published Table I) and 'calibrated'
           (config/channel_parameters_calibrated.yaml, scripts/calibrate_model.py).
--analyse  CvM distances vs the digitised Fig. 17 CDFs, Table II path-loss fits, verification of
           the GNU Radio sounder chain against the generator's ground truth; figures and tables
           in results/gnuradio/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, setup_mpl  # noqa: E402
from channel_model.path_loss import ci_overlap, fit_alpha_beta, fit_close_in  # noqa: E402
from channel_model.power_model import fspl_db  # noqa: E402
from channel_model.validation import bootstrap_cvm_null, cvm_distance, load_curve, load_samples  # noqa: E402

GR_DIR = ROOT / "gnuradio"
OUT = ROOT / "results" / "gnuradio"
RAW = OUT / "raw"
GR_PY = Path(os.environ.get("GR_PYTHON", Path.home() / "radioconda" / "python.exe"))
GRCC = GR_PY.parent / "Scripts" / "grcc.exe"
SETS = {"table1": "", "calibrated": "config/channel_parameters_calibrated.yaml"}
SET_LABEL = {"table1": "GNU Radio - Table I (published)", "calibrated": "GNU Radio - calibrated"}
N_MEAS = {"los": 21, "nlos": 17}
METRICS = [("delay", "gr_delay_spread_m", "fig17a_delay_spread", "RMS delay spread [m]"),
           ("doa", "gr_doa_spread", "fig17b_doa_spread", "DoA angular spread"),
           ("dod", "gr_dod_spread", "fig17c_dod_spread", "DoD angular spread")]
FC = 145.5e9


def compile_flowgraph(log) -> Path:
    grc = GR_DIR / "subthz_sounder_batch.grc"
    py = GR_DIR / "subthz_sounder_batch.py"
    if not py.exists() or py.stat().st_mtime < grc.stat().st_mtime:
        log.info("grcc %s", grc.name)
        subprocess.run([str(GRCC), "-o", str(GR_DIR), str(grc)], check=True, capture_output=True)
    return py


def run_dataset(py: Path, pset: str, kind: str, cond: str, n: int, workers: int, seed: int, log) -> Path:
    """Split n realizations over `workers` GNU Radio processes and merge their CSVs."""
    out = RAW / f"{pset}_{kind}_{cond}.csv"
    if out.exists() and len(pd.read_csv(out)) == n:
        log.info("cached %s (%d)", out.name, n)
        return out
    chunks = np.array_split(np.arange(n), workers)
    parts = []

    def job(k_ch):
        k, ch = k_ch
        if ch.size == 0:
            return None
        part = RAW / f"_{pset}_{kind}_{cond}_{k}.csv"
        part.unlink(missing_ok=True)
        cmd = [str(GR_PY), "-u", str(py), "--condition", cond, "--n-channels", str(ch.size),
               "--start-index", str(int(ch[0])), "--seed", str(seed),
               "--distances", "measured" if kind == "val" else "uniform",
               "--params-file", SETS[pset], "--out-csv", str(part)]
        env = {**os.environ, "SUBTHZ_REPO": str(ROOT)}
        r = subprocess.run(cmd, cwd=str(GR_DIR), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"GNU Radio run failed ({' '.join(cmd)}):\n{r.stderr[-2000:]}")
        return part

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(job, enumerate(chunks)) if p is not None]
    df = pd.concat([pd.read_csv(p) for p in parts]).sort_values("rid")
    df.to_csv(out, index=False)
    for p in parts:
        p.unlink()
    log.info("%s: %d channels in %.0f s (GNU Radio, %d processes)", out.name, len(df), time.perf_counter() - t0, workers)
    return out


# --------------------------------------------------------------------------- analysis
def bench() -> dict:
    return yaml.safe_load((ROOT / "config" / "channel_parameters.yaml").read_text())["published_benchmarks"]


def load(pset: str, kind: str, cond: str) -> pd.DataFrame | None:
    f = RAW / f"{pset}_{kind}_{cond}.csv"
    return pd.read_csv(f) if f.exists() else None


def cvm_rows(sets: list[str]) -> pd.DataFrame:
    pub = bench()["cvm"]
    rng = np.random.default_rng(5)
    rows = []
    for cond in ("los", "nlos"):
        for key, col, fig, _ in METRICS:
            meas = load_samples(fig, f"{cond}_measured")
            syn = load_samples(fig, f"{cond}_paper_synthetic")
            row = {"metric": key, "condition": cond, "paper_published_cvm": pub[key][cond],
                   "paper_recomputed_cvm": cvm_distance(meas, syn),
                   "median_measured": float(np.median(meas)), "median_paper_synthetic": float(np.median(syn))}
            for s in sets:
                df = load(s, "val", cond)
                if df is None:
                    continue
                v = df[col].dropna().to_numpy()
                null = bootstrap_cvm_null(meas.size, v.size, 1000, rng)
                row.update({f"{s}_n": v.size, f"{s}_cvm_vs_measured": cvm_distance(v, meas),
                            f"{s}_cvm_vs_paper_synthetic": cvm_distance(v, syn), f"{s}_median": float(np.median(v)),
                            f"{s}_null95": float(np.quantile(null, 0.95))})
            rows.append(row)
    return pd.DataFrame(rows)


def pathloss_rows(sets: list[str], n_rep: int = 200) -> pd.DataFrame:
    pub = bench()["path_loss_table_ii"]
    rng = np.random.default_rng(29)
    rows = []
    for cond in ("los", "nlos"):
        for mode, col in (("omni", "gr_pg_omni_db"), ("maxdir", "gr_pg_maxdir_db")):
            key = f"{cond}_{mode}"
            ab_pub, ci_pub = pub["alpha_beta"][key], pub["close_in"][key]
            for s in sets:
                df = load(s, "pl", cond)
                if df is None:
                    continue
                d, pl = df.d_m.to_numpy(), -df[col].to_numpy()
                full_ab, full_ci = fit_alpha_beta(d, pl), fit_close_in(d, pl, FC)
                reps = []
                for _ in range(n_rep):     # the paper's experiment: 50 channels per fit
                    i = rng.choice(d.size, 50, replace=False)
                    ab, ci = fit_alpha_beta(d[i], pl[i]), fit_close_in(d[i], pl[i], FC)
                    reps.append((ab, ci))
                for par, get, pv, pci, ref in (
                        ("alpha", lambda ab, ci: (ab.alpha, ab.alpha_ci), ab_pub["alpha"], ab_pub["alpha_ci"], ab_pub["alpha_ref"]),
                        ("beta", lambda ab, ci: (ab.beta, ab.beta_ci), ab_pub["beta"], ab_pub["beta_ci"], ab_pub["beta_ref"]),
                        ("sigma", lambda ab, ci: (ab.sigma, ab.sigma_ci), ab_pub["sigma"], ab_pub["sigma_ci"], ab_pub["sigma_ref"]),
                        ("n_CI", lambda ab, ci: (ci.n_exp, ci.n_ci), ci_pub["n_fit"], ci_pub["n_fit_ci"], ci_pub["n_mea"])):
                    vals = np.array([get(ab, ci)[0] for ab, ci in reps])
                    ov = np.mean([ci_overlap(tuple(get(ab, ci)[1]), tuple(pci)) for ab, ci in reps])
                    rows.append({"case": key, "set": s, "parameter": par, "measured_[21]": ref, "paper_synthetic": pv,
                                 "paper_ci_lo": pci[0], "paper_ci_hi": pci[1],
                                 "ours_all_channels": float(get(full_ab, full_ci)[0]),
                                 "ours_median_50ch": float(np.median(vals)), "ours_p2.5": float(np.quantile(vals, 0.025)),
                                 "ours_p97.5": float(np.quantile(vals, 0.975)), "ci_overlap_share": float(ov)})
    return pd.DataFrame(rows)


def verify_rows(sets: list[str]) -> pd.DataFrame:
    rows = []
    for s in sets:
        for kind in ("val", "pl"):
            for cond in ("los", "nlos"):
                df = load(s, kind, cond)
                if df is None:
                    continue
                for k in ("delay_spread_m", "doa_spread", "dod_spread", "pg_omni_db", "pg_maxdir_db"):
                    g, t = df[f"gr_{k}"].to_numpy(), df[f"truth_{k}"].to_numpy()
                    err = g - t if k.startswith("pg") else (g - t) / np.where(t != 0, t, np.nan)
                    rows.append({"set": s, "data": kind, "condition": cond, "metric": k, "n": len(df),
                                 "median_abs_err": float(np.nanmedian(np.abs(err))),
                                 "p95_abs_err": float(np.nanquantile(np.abs(err), 0.95)),
                                 "unit": "dB" if k.startswith("pg") else "relative"})
    return pd.DataFrame(rows)


def figures(sets: list[str], cvm: pd.DataFrame) -> None:
    plt = setup_mpl()
    col = {"table1": "#ff7f0e", "calibrated": "#2ca02c"}
    # --- Fig. 17 overlay
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
    for i, cond in enumerate(("los", "nlos")):
        for j, (key, c, figname, xl) in enumerate(METRICS):
            ax = axs[i, j]
            for role, lab, sty in (("measured", "measured (paper, digitized)", dict(color="k", lw=2)),
                                   ("paper_synthetic", "paper synthetic (digitized)", dict(color="#7f7f7f", lw=2, ls="--"))):
                x, y = load_curve(figname, f"{cond}_{role}")
                ax.step(x, y, where="post", label=lab, **sty)
            for s in sets:
                df = load(s, "val", cond)
                if df is None:
                    continue
                v = np.sort(df[c].dropna().to_numpy())
                r = cvm[(cvm.metric == key) & (cvm.condition == cond)].iloc[0]
                ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=col[s], lw=1.6,
                        label=f"{SET_LABEL[s]} (w2 vs meas. {r[f'{s}_cvm_vs_measured']:.3f})")
            pub = r["paper_published_cvm"]
            ax.set_title(f"{cond.upper()} - {xl}  (paper w2 = {pub})", fontsize=9)
            ax.set_xlabel(xl)
            ax.set_ylabel("CDF")
            ax.legend(fontsize=6.3, loc="lower right" if cond == "los" else "upper left")
    fig.suptitle("Fig. 17 reproduced with the GNU Radio channel sounder", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gr_fig17_validation.png")
    plt.close(fig)

    # --- path loss
    pub = bench()["path_loss_table_ii"]["alpha_beta"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    dd = np.logspace(0, 2, 100)
    for ax, cond in zip(axs, ("los", "nlos")):
        p = pub[f"{cond}_omni"]
        ax.plot(dd, p["alpha"] + 10 * p["beta"] * np.log10(dd), "k-", lw=2, label=f"paper synthetic: {p['alpha']} + 10*{p['beta']} log d")
        ax.plot(dd, p["alpha_ref"] + 10 * p["beta_ref"] * np.log10(dd), "k:", lw=1.5,
                label=f"measured [21]: {p['alpha_ref']} + 10*{p['beta_ref']} log d")
        ax.plot(dd, fspl_db(dd, FC), color="#999", lw=1, label="free space")
        for s in sets:
            df = load(s, "pl", cond)
            if df is None:
                continue
            d, pl = df.d_m.to_numpy(), -df.gr_pg_omni_db.to_numpy()
            ab = fit_alpha_beta(d, pl)
            ax.scatter(d, pl, s=5, alpha=0.35, color=col[s])
            ax.plot(dd, ab.alpha + 10 * ab.beta * np.log10(dd), color=col[s], lw=1.8,
                    label=f"{SET_LABEL[s]}: {ab.alpha:.2f} + 10*{ab.beta:.2f} log d")
        ax.set_xscale("log")
        ax.set_xlabel("Tx-Rx distance d [m]")
        ax.set_ylabel("omni path loss [dB]")
        ax.set_title(f"{cond.upper()} omni path loss (Table II)")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "gr_pathloss.png")
    plt.close(fig)

    # --- sounder verification
    s = sets[0]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (k, lab) in zip(axs, [("delay_spread_m", "delay spread [m]"), ("doa_spread", "DoA spread"),
                                  ("dod_spread", "DoD spread"), ("pg_omni_db", "omni path gain [dB]")]):
        for cond, c in (("los", "#1f77b4"), ("nlos", "#d62728")):
            df = load(s, "val", cond)
            if df is None:
                continue
            ax.scatter(df[f"truth_{k}"], df[f"gr_{k}"], s=6, color=c, alpha=0.5, label=cond.upper())
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"generator ADPS (truth): {lab}")
        ax.set_ylabel("GNU Radio sounder output")
        ax.legend(fontsize=7)
    fig.suptitle("GNU Radio VNA sounder chain (noise, IFFT, |.|^2, 35 dB taps) recovers the channel metrics", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gr_sounder_verification.png")
    plt.close(fig)

    # --- medians bar chart
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, c, figname, xl) in zip(axs, METRICS):
        sub = cvm[cvm.metric == key].set_index("condition")
        labels = ["measured", "paper synthetic"] + [SET_LABEL[s].replace("GNU Radio - ", "GR ") for s in sets]
        x = np.arange(2)
        w = 0.8 / len(labels)
        vals = [sub.median_measured, sub.median_paper_synthetic] + [sub[f"{s}_median"] for s in sets]
        colors = ["k", "#7f7f7f"] + [col[s] for s in sets]
        for k, (lab, v, cc) in enumerate(zip(labels, vals, colors)):
            ax.bar(x + (k - (len(labels) - 1) / 2) * w, v.loc[["los", "nlos"]], w, label=lab, color=cc)
        ax.set_xticks(x, ["LoS", "NLoS"])
        ax.set_title(f"median {xl}")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "gr_median_comparison.png")
    plt.close(fig)


def analyse(sets: list[str], log) -> None:
    cvm = cvm_rows(sets)
    pl = pathloss_rows(sets)
    ver = verify_rows(sets)
    cvm.to_csv(OUT / "gr_cvm.csv", index=False, float_format="%.5g")
    pl.to_csv(OUT / "gr_pathloss.csv", index=False, float_format="%.4g")
    ver.to_csv(OUT / "gr_sounder_verification.csv", index=False, float_format="%.4g")
    counts = {}
    for s in sets:
        for cond in ("los", "nlos"):
            df = load(s, "val", cond)
            if df is not None:
                counts[f"{s}_{cond}"] = {"mean_observable_clusters": float(df.truth_n_visible.mean()),
                                         "mean_drawn_clusters": float(df.truth_n_clusters.mean())}
    (OUT / "gr_summary.json").write_text(json.dumps({"cvm": cvm.to_dict("records"), "pathloss": pl.to_dict("records"),
                                                     "verification": ver.to_dict("records"), "cluster_counts": counts},
                                                    indent=2))
    figures(sets, cvm)
    log.info("CvM:\n%s", cvm.round(4).to_string())
    log.info("Path loss (omni):\n%s", pl[pl.case.str.endswith("omni")].round(3).to_string())
    log.info("Sounder verification:\n%s", ver.groupby(["set", "metric"]).median_abs_err.max().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--sets", nargs="+", default=["table1", "calibrated"], choices=list(SETS))
    ap.add_argument("--reps", type=int, default=10, help="channels per measured link distance (Fig. 17 sets)")
    ap.add_argument("--n-pl", type=int, default=300, help="channels at d ~ U(1,100) m per condition (Table II)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    log = get_logger("run_gnuradio")
    if a.run:
        py = compile_flowgraph(log)
        for s in a.sets:
            if SETS[s] and not (ROOT / SETS[s]).exists():
                raise SystemExit(f"{SETS[s]} missing - run scripts/calibrate_model.py --search first")
            for cond in ("los", "nlos"):
                run_dataset(py, s, "val", cond, a.reps * N_MEAS[cond], a.workers, a.seed, log)
                run_dataset(py, s, "pl", cond, a.n_pl, a.workers, a.seed, log)
    if a.analyse:
        analyse(a.sets, log)


if __name__ == "__main__":
    main()
