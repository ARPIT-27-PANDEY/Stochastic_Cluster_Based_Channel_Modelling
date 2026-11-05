"""Compact figures for the slide deck (results/gnuradio/slide_*.png).

    python scripts/make_slide_figures.py

Uses the GNU Radio sounder output of scripts/run_gnuradio.py (set 'calibrated' unless --set is given)
and the digitised paper curves.  Every curve is labelled with its source.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, setup_mpl  # noqa: E402
from channel_model.path_loss import fit_alpha_beta  # noqa: E402
from channel_model.validation import cvm_distance, load_curve, load_samples  # noqa: E402

GRD = ROOT / "results" / "gnuradio"
METRICS = [("gr_delay_spread_m", "fig17a_delay_spread", "RMS delay spread [m]", "delay"),
           ("gr_doa_spread", "fig17b_doa_spread", "DoA angular spread", "doa"),
           ("gr_dod_spread", "fig17c_dod_spread", "DoD angular spread", "dod")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="calibrated")
    a = ap.parse_args()
    plt = setup_mpl()
    import matplotlib.pyplot as plt  # noqa: F811
    lab = "GNU Radio sounder (calibrated)" if a.set == "calibrated" else "GNU Radio sounder (Table I)"
    pub = yaml.safe_load((ROOT / "config" / "channel_parameters.yaml").read_text())["published_benchmarks"]

    # ---- Fig. 17
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
    for i, cond in enumerate(("los", "nlos")):
        df = pd.read_csv(GRD / "raw" / f"{a.set}_val_{cond}.csv")
        for j, (col, fname, xl, key) in enumerate(METRICS):
            ax = axs[i, j]
            x, y = load_curve(fname, f"{cond}_measured")
            ax.step(x, y, where="post", color="k", lw=2, label="measured (paper, digitized)")
            x, y = load_curve(fname, f"{cond}_paper_synthetic")
            ax.step(x, y, where="post", color="#7f7f7f", lw=2, ls="--", label="paper synthetic (digitized)")
            v = np.sort(df[col].dropna().to_numpy())
            w2 = cvm_distance(v, load_samples(fname, f"{cond}_measured"))
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color="#2ca02c", lw=1.8,
                    label=f"{lab}")
            ax.set_title(f"{cond.upper()} - {xl}\nomega2 vs measured: paper {pub['cvm'][key][cond]}, ours {w2:.3f}", fontsize=9)
            ax.set_xlabel(xl)
            ax.set_ylabel("CDF")
            if cond == "los" and key == "delay":
                ax.set_xlim(0, 40)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(GRD / "slide_fig17.png", dpi=200)
    plt.close(fig)

    # ---- path loss (Table II)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
    dd = np.logspace(0, 2, 100)
    for ax, cond in zip(axs, ("los", "nlos")):
        df = pd.read_csv(GRD / "raw" / f"{a.set}_pl_{cond}.csv")
        d, pl = df.d_m.to_numpy(), -df.gr_pg_omni_db.to_numpy()
        ab = fit_alpha_beta(d, pl)
        p = pub["path_loss_table_ii"]["alpha_beta"][f"{cond}_omni"]
        ax.scatter(d, pl, s=6, color="#2ca02c", alpha=0.35, label=f"{lab}, {d.size} channels")
        ax.plot(dd, ab.alpha + 10 * ab.beta * np.log10(dd), color="#2ca02c", lw=2,
                label=f"fit: alpha {ab.alpha:.1f} dB, beta {ab.beta:.2f}")
        ax.plot(dd, p["alpha"] + 10 * p["beta"] * np.log10(dd), "k--", lw=2,
                label=f"paper Table II: alpha {p['alpha']} dB, beta {p['beta']}")
        ax.set_xscale("log")
        ax.set_xlabel("Tx-Rx distance [m]")
        ax.set_ylabel("omni path loss [dB]")
        ax.set_title(f"{cond.upper()} omni path loss", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(GRD / "slide_pathloss.png", dpi=200)
    plt.close(fig)
    print("saved", GRD / "slide_fig17.png", GRD / "slide_pathloss.png")


if __name__ == "__main__":
    main()
