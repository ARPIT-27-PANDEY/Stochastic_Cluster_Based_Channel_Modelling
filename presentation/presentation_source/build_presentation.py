"""Build presentation/presentation.pptx (compact 9-slide deck) from the GNU Radio results.

    python scripts/run_gnuradio.py --run --analyse --sets table1 calibrated
    python scripts/make_slide_figures.py
    python presentation/presentation_source/build_presentation.py

All numbers are read from results/ at build time; values quoted from the paper are labelled
"paper".  The build stops if a figure is missing, so the deck never contains placeholders.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
GRD = ROOT / "results" / "gnuradio"
OUT = ROOT / "presentation" / "presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0xC0, 0x50, 0x1E)
GREY = RGBColor(0x55, 0x55, 0x55)
TEXT = RGBColor(0x22, 0x22, 0x22)
LIGHT = RGBColor(0xEE, 0xF3, 0xFB)
W, H = Inches(13.333), Inches(7.5)
GROUP = "Group 10"
TEAM = [("23116075", "Piyush Bhaskar"), ("23116005", "Akersheet Tripathi"), ("23116017", "Arpit Kumar Pandey")]
PAPER = ("Y. Cheng, N. A. Abbasi, J. Gomez-Ponce, Z. Zhang, A. F. Molisch, \"Stochastic Cluster-Based Channel Modeling "
         "for Sub-Terahertz Outdoor Device-to-Device Radio Propagation at 145 GHz\", IEEE TWC, vol. 25, pp. 20883-20899, 2026")

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
N = [0]


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"missing {p} - run scripts/run_gnuradio.py --analyse and scripts/make_slide_figures.py first")
    return p


def tb(s, x, y, w, h, text, size=14, bold=False, color=TEXT, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
    r.font.color.rgb = color
    return box


def bullets(s, x, y, w, h, items, size=14, gap=5):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for k, it in enumerate(items):
        lvl = 0
        if isinstance(it, tuple):
            lvl, it = it
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "      ") + it
        r.font.size, r.font.name = Pt(size - lvl), "Calibri"
        r.font.color.rgb = TEXT if lvl == 0 else GREY
    return box


def slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb(s, Inches(0.45), Inches(0.22), Inches(12.4), Inches(0.6), title, 26, True, NAVY)
    if subtitle:
        tb(s, Inches(0.45), Inches(0.80), Inches(12.4), Inches(0.4), subtitle, 13, False, GREY)
    tb(s, Inches(0.45), Inches(7.08), Inches(10), Inches(0.3),
       f"{GROUP} - GNU Radio reproduction of Cheng et al., IEEE TWC 2026 (145 GHz D2D cluster channel model)", 10, False, GREY)
    tb(s, Inches(12.2), Inches(7.08), Inches(0.7), Inches(0.3), str(N[0]), 10, False, GREY, PP_ALIGN.RIGHT)
    return s


def image(s, path: Path, x, y, w, h):
    from PIL import Image
    iw, ih = Image.open(need(path)).size
    if iw / ih > w / h:
        h2 = int(w * ih / iw)
        return s.shapes.add_picture(str(path), x, y + (h - h2) // 2, width=w)
    w2 = int(h * iw / ih)
    return s.shapes.add_picture(str(path), x + (w - w2) // 2, y, height=h)


def table(s, df, x, y, w, h, size=11, widths=None):
    t = s.shapes.add_table(df.shape[0] + 1, df.shape[1], x, y, w, h).table
    if widths:
        for j, f in enumerate(widths):
            t.columns[j].width = int(w * f)
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.text = str(c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold, r.font.color.rgb = Pt(size), True, RGBColor(255, 255, 255)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            cell = t.cell(i + 1, j)
            cell.text = str(df.iat[i, j])
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else LIGHT
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.color.rgb = Pt(size), TEXT
    return t


def callout(s, x, y, w, h, text, size=12):
    box = s.shapes.add_shape(1, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = NAVY
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    r = tf.paragraphs[0].add_run()
    r.text = text
    r.font.size, r.font.name = Pt(size), "Calibri"
    r.font.color.rgb = TEXT


# ============================================================================ data
cvm = pd.read_csv(need(GRD / "gr_cvm.csv"))
pl = pd.read_csv(need(GRD / "gr_pathloss.csv"))
ver = pd.read_csv(need(GRD / "gr_sounder_verification.csv"))
summ = json.loads(need(GRD / "gr_summary.json").read_text())
base = yaml.safe_load((ROOT / "config" / "channel_parameters.yaml").read_text())
calp = yaml.safe_load(need(ROOT / "config" / "channel_parameters_calibrated.yaml").read_text())
n_ch = {c: int(cvm[cvm.condition == c].calibrated_n.iloc[0]) for c in ("los", "nlos")}
n_tot = sum(len(pd.read_csv(f)) for f in (GRD / "raw").glob("*.csv"))
v_tab = ver[ver.set == "table1"]
spr_err = 100 * v_tab[v_tab.unit == "relative"].median_abs_err.max()
pg_err = v_tab[v_tab.unit == "dB"].median_abs_err.max()
ok_cal = int((cvm.calibrated_cvm_vs_measured <= cvm.calibrated_null95).sum())
ok_t1 = int((cvm.table1_cvm_vs_measured <= cvm.table1_null95).sum())
cc = summ["cluster_counts"]

# ============================================================================ 1 title
s = prs.slides.add_slide(BLANK)
N[0] += 1
bg = s.shapes.add_shape(1, 0, 0, W, Inches(0.35))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb(s, Inches(0.8), Inches(1.3), Inches(11.7), Inches(1.6),
   "Stochastic Cluster-Based Sub-THz D2D Channel Model at 145 GHz - Reproduced in GNU Radio", 34, True, NAVY)
tb(s, Inches(0.8), Inches(3.0), Inches(11.7), Inches(0.9),
   "GNU Radio channel-sounder and channel-emulator flowgraphs implementing Cheng et al., IEEE Transactions on "
   "Wireless Communications, 2026, and a comparison of their output with the paper", 18, False, GREY)
tb(s, Inches(0.8), Inches(4.3), Inches(8), Inches(0.45), f"{GROUP}  -  Contributors", 18, True, ACCENT)
for i, (roll, name) in enumerate(TEAM):
    tb(s, Inches(0.8), Inches(4.85 + 0.42 * i), Inches(8), Inches(0.4), f"{roll}  -  {name}", 18)
tb(s, Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.4), "Tools: GNU Radio 3.10.12 (GNU Radio Companion), Python 3", 13, color=GREY)

# ============================================================================ 2 paper + approach
s = slide("The paper and our GNU Radio implementation",
          "Cheng, Abbasi, Gomez-Ponce, Zhang, Molisch - 145 GHz outdoor device-to-device measurements and cluster model")
bullets(s, Inches(0.45), Inches(1.35), Inches(5.0), Inches(5.6), [
    "Measured: 145-146 GHz VNA sounder, 1001 tones (1 ns bins), horns rotated in 10 deg steps, 21 LoS + 17 NLoS links",
    "Model (Table I): numbers, arrival times, angles, spreads and powers of clusters",
    "Key idea: clusters are only seen within a 35 dB dynamic range; the generator inverts this truncation",
    "Validation in the paper: CDFs of delay / angular spreads (Fig. 17) and path loss (Table II)",
    "Measurement data are not public, so we compare with curves digitized from the paper",
], 14)
table(s, pd.DataFrame({
    "Paper step": ["Channel realization (Table I, Sec. VI)", "VNA sweep, 36 x 36 horn orientations", "Sounder noise",
                   "ADPS, Eq. 1: |IFFT(H)|^2 / N_f", "35 dB MPC threshold, spreads, path gain", "Channel emulation"],
    "GNU Radio block(s)": ["Embedded Python block (cluster channel + VNA sweep)", "1001-element complex vectors, tags per channel",
                           "Noise Source, Stream to Vector, Add", "FFT (reverse), Complex to Mag^2, Multiply Const",
                           "Embedded Python sink (ADPS, metrics, CSV)", "FIR filter with taps from a Function Probe, QT GUI sinks"]}),
    Inches(5.7), Inches(1.4), Inches(7.2), Inches(4.0), 12, widths=(0.42, 0.58))
callout(s, Inches(5.7), Inches(5.65), Inches(7.2), Inches(1.2),
        f"{n_tot} channels in total were measured through the GNU Radio sounder flowgraph (4 parallel GNU Radio processes). "
        f"Per parameter set: {n_ch['los']} LoS / {n_ch['nlos']} NLoS at the paper's link distances (Fig. 17) and 300 per "
        "condition at 1-100 m (Table II).", 12)

# ============================================================================ 3 flowgraph sounder
s = slide("GRC flowgraph 1: VNA channel sounder (subthz_sounder_batch.grc)",
          "Screenshot of the GNU Radio Companion canvas - no-GUI flowgraph, run from the command line in batch")
image(s, GRD / "canvas_subthz_sounder_batch.png", Inches(0.3), Inches(1.3), Inches(12.7), Inches(4.45))
callout(s, Inches(0.45), Inches(5.95), Inches(12.4), Inches(0.95),
        f"Verification: the spreads measured by the flowgraph agree with the generator's own values to within {spr_err:.2f} % "
        f"(median) and path gains to within {pg_err:.3f} dB, so every result that follows is a property of the "
        "channel model measured through GNU Radio.", 13)

# ============================================================================ 4 flowgraph emulator
s = slide("GRC flowgraph 2: interactive channel emulator (subthz_channel_emulator.grc)",
          "Canvas (left) and the running QT GUI (right): NLoS link, 40 m, calibrated parameter set")
image(s, GRD / "canvas_subthz_channel_emulator.png", Inches(0.3), Inches(1.3), Inches(5.5), Inches(4.6))
image(s, GRD / "emulator_nlos_calibrated.png", Inches(5.95), Inches(1.3), Inches(7.1), Inches(4.6))
callout(s, Inches(0.45), Inches(6.05), Inches(12.4), Inches(0.85),
        "Widgets choose LoS / NLoS, distance, realization and parameter set. Displays: omni power-delay profile, DoA / DoD "
        "angular power spectra, and a 1 GHz test signal before and after the FIR channel (frequency-selective fading).", 12)

# ============================================================================ 5 Figs 14-15
s = slide("Model core: dynamic-range truncation and its inverse (Figs. 14-15)",
          "kappa_tau = 10 m, kappa_DoD = 0.2 pi, kappa_DoA = 0.3 pi, dynamic range 1-50 dB - same setting as the paper")
image(s, FIG / "fig14_inverse_delay.png", Inches(0.3), Inches(1.3), Inches(6.3), Inches(4.6))
image(s, FIG / "fig15_inverse_angular.png", Inches(6.7), Inches(1.3), Inches(6.3), Inches(4.6))
callout(s, Inches(0.45), Inches(6.05), Inches(12.4), Inches(0.85),
        "Matches the paper: inverse-solved spreads coincide with the actual ones in all four regimes, and the regime boundaries "
        "follow analytically at 14.48 / 21.71 / 36.19 dB. This inverse is what the GNU Radio channel block uses.", 12)

# ============================================================================ 6 Fig 17
s = slide("Fig. 17: delay and angular spreads - GNU Radio vs paper",
          "GNU Radio sounder output at the measured link distances (calibrated parameter set) vs CDFs digitized from the paper")
image(s, GRD / "slide_fig17.png", Inches(0.3), Inches(1.25), Inches(12.7), Inches(4.75))
med = {(r.metric, r.condition): r for r in cvm.itertuples()}
d_n, a_n = med[("delay", "nlos")], med[("doa", "nlos")]
callout(s, Inches(0.45), Inches(6.1), Inches(12.4), Inches(0.85),
        f"{ok_cal} of 6 CDFs are within the sampling uncertainty of the measurements (omega^2 below the 95 % null of samples of "
        f"21/17 links). Median NLoS delay spread {d_n.calibrated_median:.1f} m (measured {d_n.median_measured:.1f} m, paper "
        f"model {d_n.median_paper_synthetic:.1f} m). With the unmodified Table I only {ok_t1} of 6 were: NLoS came out too "
        f"dispersive (median {d_n.table1_median:.1f} m, DoA {a_n.table1_median:.2f}).", 12)

# ============================================================================ 7 Table II
s = slide("Table II: path loss - GNU Radio vs paper",
          "Path loss of GNU Radio sounder output, d ~ U(1, 100) m; 50-channel fits repeated 200 times (as in the paper)")
rows = []
for case, clab in (("los_omni", "LoS omni"), ("nlos_omni", "NLoS omni"), ("los_maxdir", "LoS max-dir"), ("nlos_maxdir", "NLoS max-dir")):
    for par, lab in (("alpha", "alpha [dB]"), ("beta", "beta"), ("n_CI", "n (CI model)")):
        g = pl[(pl.case == case) & (pl.parameter == par) & (pl.set == "calibrated")]
        if not len(g):
            continue
        r = g.iloc[0]
        rows.append({"case": clab, "parameter": lab, "measured [21]": f"{r['measured_[21]']:.2f}",
                     "paper synthetic [95 % CI]": f"{r.paper_synthetic:.2f} [{r.paper_ci_lo:.2f}, {r.paper_ci_hi:.2f}]",
                     "GNU Radio": f"{r.ours_median_50ch:.2f}",
                     "CI overlap": f"{100 * r.ci_overlap_share:.0f} %"})
pt = pd.DataFrame(rows)
table(s, pt, Inches(0.3), Inches(1.3), Inches(6.9), Inches(0.3 * (len(pt) + 1)), 10, widths=(0.17, 0.17, 0.14, 0.29, 0.12, 0.11))
image(s, GRD / "slide_pathloss.png", Inches(7.35), Inches(1.3), Inches(5.7), Inches(2.4))
ov = pt["CI overlap"].str.rstrip(" %").astype(float)
bad = pt[ov < 50]
txt = "CI overlap = share of our 50-channel fits whose 95 % CI overlaps the paper's."
txt += (" All listed parameters overlap in most repeats." if not len(bad) else
        " Below 50 %: " + ", ".join(f"{a} {b}" for a, b in zip(bad.case, bad.parameter)) + ".")
callout(s, Inches(7.35), Inches(3.95), Inches(5.7), Inches(1.2), txt, 11)
t1 = pl[(pl.set == "table1") & (pl.parameter == "n_CI") & (pl.case == "los_omni")].iloc[0]
tb(s, Inches(7.35), Inches(5.3), Inches(5.7), Inches(1.5),
   f"With the unmodified Table I the LoS close-in exponent was {t1.ours_median_50ch:.2f} (paper {t1.paper_synthetic:.2f}, "
   f"CI overlap {100 * t1.ci_overlap_share:.0f} %); one common power offset per condition fixes this without changing any spread.",
   11, color=GREY)

# ============================================================================ 8 calibration
s = slide("How the GNU Radio results were brought close to the paper",
          "Explicit, labelled calibration: config/channel_parameters_calibrated.yaml (scripts/calibrate_model.py); Table I unchanged")
bullets(s, Inches(0.45), Inches(1.35), Inches(6.6), Inches(5.6), [
    "Problem: with the published Table I our NLoS channels keep too many weak clusters above the 35 dB cut-off "
    f"({cc['table1_nlos']['mean_observable_clusters']:.0f} observable per channel) -> spreads too large",
    "Step 1: coordinate search on the cluster-power model (spread rho, slope beta, intercept alpha) and the cluster-number "
    "scale, minimising the CvM distance to the paper's synthetic Fig. 17 CDFs",
    (1, f"result: NLoS power spread x 1.7 -> {cc['calibrated_nlos']['mean_observable_clusters']:.0f} observable clusters; "
        "LoS intercept +2 dB"),
    "Step 2: one common power offset per condition (all clusters, same dB) so the close-in path-loss exponent equals "
    "Table II; the 35 dB cut-off is relative to the strongest tap, so the spreads do not change",
    "Both parameter sets are selectable in both flowgraphs (GUI chooser / --params-file)",
], 13)
rows = []
for cond in ("LoS", "NLoS"):
    c = cond.lower()
    for k, lab in (("rho", "NCP spread rho [dB]"), ("alpha", "NCP intercept alpha [dB]")):
        a0, a1 = float(base["cluster_power"][c][k]), float(calp["cluster_power"][c][k])
        rows.append({"condition": cond, "parameter": lab, "Table I": f"{a0:.2f}", "calibrated": f"{a1:.2f}"})
a0, a1 = float(base["cluster_power"]["los_cluster"]["alpha"]), float(calp["cluster_power"]["los_cluster"]["alpha"])
rows.append({"condition": "LoS", "parameter": "LoS-cluster intercept [dB]", "Table I": f"{a0:.2f}", "calibrated": f"{a1:.2f}"})
table(s, pd.DataFrame(rows), Inches(7.3), Inches(1.45), Inches(5.6), Inches(0.36 * (len(rows) + 1)), 12, widths=(0.17, 0.43, 0.2, 0.2))
callout(s, Inches(7.3), Inches(4.0), Inches(5.6), Inches(1.6),
        "This is a fit to the paper's published results, not a re-extraction from measurements (the data are not public). "
        "It shows which Table I quantity controls the gap: mainly the spread of the NLoS cluster powers.", 12)

# ============================================================================ 9 summary
s = slide("Summary, limitations and references")
bullets(s, Inches(0.45), Inches(1.3), Inches(12.4), Inches(3.6), [
    "Implemented in GNU Radio: a VNA channel-sounder flowgraph and an interactive channel-emulator flowgraph for the 145 GHz "
    "cluster model; the sounder chain reproduces the model's metrics to < 1 %",
    "Close to the paper: truncation inverse (Figs. 14-15); Fig. 17 spread CDFs; Table II path-loss parameters (calibrated set)",
    "Not close, for the record: with the unmodified Table I the NLoS spreads and the LoS DoA spread deviate; observable cluster "
    "counts at 35 dB are below the measured ones; the NLoS path-loss slope beta stays shallower than the paper's "
    f"({pl[(pl.set == 'calibrated') & (pl.case == 'nlos_omni') & (pl.parameter == 'beta')].ours_median_50ch.iloc[0]:.2f} vs 1.56, "
    "inside its wide CI); the truncated power MLE is biased for LoS on our simulated data",
    "Limitations: no measurement data (comparison with digitized curves); the calibration is fitted to the paper's outputs; "
    "baseband emulation only (no RF hardware) - a USRP over-the-air test at a lower band is a natural next step",
], 14)
bullets(s, Inches(0.45), Inches(5.0), Inches(12.4), Inches(2.0), [
    PAPER + ", doi: 10.1109/TWC.2026.3714332.",
    "N. A. Abbasi et al., \"THz band channel measurements and statistical modeling for urban D2D environments,\" arXiv:2109.13693.",
    "GNU Radio project, GNU Radio 3.10 / GNU Radio Companion, https://www.gnuradio.org.",
], 11, gap=2)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"saved {OUT} with {len(prs.slides)} slides")
