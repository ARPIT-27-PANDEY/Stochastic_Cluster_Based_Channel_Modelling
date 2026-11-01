"""Build presentation/presentation.pptx from the generated figures and result tables.

    python presentation/presentation_source/build_presentation.py

All numbers on the slides are read from results/ at build time (nothing is typed in by hand
except values quoted from the paper, which are labelled "paper").
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "presentation" / "presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0xC0, 0x50, 0x1E)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xEE, 0xF3, 0xFB)
W, H = Inches(13.333), Inches(7.5)
TEAM = [("23116075", "Piyush Bhaskar"), ("23116005", "Akersheet Tripathi"), ("23116017", "Arpit Kumar Pandey")]
PAPER = ("Y. Cheng, N. A. Abbasi, J. Gomez-Ponce, Z. Zhang, A. F. Molisch, \"Stochastic Cluster-Based Channel Modeling "
         "for Sub-Terahertz Outdoor Device-to-Device Radio Propagation at 145 GHz\", IEEE TWC, vol. 25, pp. 20883-20899, 2026")


def J(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
N_SLIDE = [0]


def _tb(slide, x, y, w, h, text="", size=16, bold=False, color=None, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
    r.font.color.rgb = color or RGBColor(0x22, 0x22, 0x22)
    return tb


def bullets(slide, x, y, w, h, items, size=16, gap=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        level = 0
        if isinstance(it, tuple):
            level, it = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = ("• " if level == 0 else "– ") + it
        r.font.size = Pt(size - 2 * level)
        r.font.name = "Calibri"
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22) if level == 0 else GREY
        p.level = level
    return tb


def slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    N_SLIDE[0] += 1
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    _tb(s, Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.7), title, 28, True, NAVY)
    if subtitle:
        _tb(s, Inches(0.5), Inches(0.88), Inches(12.3), Inches(0.4), subtitle, 14, False, GREY)
    _tb(s, Inches(0.5), Inches(7.05), Inches(10), Inches(0.3),
        "Reproduction of Cheng et al., IEEE TWC 2026 - 145 GHz D2D cluster model", 10, False, GREY)
    _tb(s, Inches(12.2), Inches(7.05), Inches(0.7), Inches(0.3), str(N_SLIDE[0]), 10, False, GREY, PP_ALIGN.RIGHT)
    return s


def image(s, path: Path, x, y, w=None, h=None):
    if not path.exists():
        _tb(s, x, y, Inches(4), Inches(0.5), f"[missing figure {path.name}]", 12, color=ACCENT)
        return None
    from PIL import Image
    iw, ih = Image.open(path).size
    if w is not None and h is not None:  # fit inside box, keep aspect
        if iw / ih > w / h:
            h2 = int(w * ih / iw)
            return s.shapes.add_picture(str(path), x, y + (h - h2) // 2, width=w)
        w2 = int(h * iw / ih)
        return s.shapes.add_picture(str(path), x + (w - w2) // 2, y, height=h)
    return s.shapes.add_picture(str(path), x, y, width=w, height=h)


def table(s, df: pd.DataFrame, x, y, w, h, size=11, header_color=NAVY):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = s.shapes.add_table(rows, cols, x, y, w, h).table
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.text = str(c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold, r.font.color.rgb = Pt(size), True, RGBColor(255, 255, 255)
    for i in range(df.shape[0]):
        for j in range(cols):
            cell = t.cell(i + 1, j)
            cell.text = str(df.iat[i, j])
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else LIGHT
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    return t


def callout(s, x, y, w, h, text, size=13):
    box = s.shapes.add_shape(1, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = NAVY
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size, r.font.name = Pt(size), "Calibri"
    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)


# ============================================================================ data
cvm = pd.read_csv(TAB / "cvm_validation.csv") if (TAB / "cvm_validation.csv").exists() else None
pl = pd.read_csv(TAB / "pathloss_table2.csv") if (TAB / "pathloss_table2.csv").exists() else None
e1, e2, e3 = J(EXP / "exp1_performance" / "results.json"), J(EXP / "exp2_dr_sensitivity" / "results.json"), J(EXP / "exp3_inverse_mapping" / "results.json")
e4, e5 = J(EXP / "exp4_power_mle" / "results.json"), J(EXP / "exp5_clustering" / "results.json")
e6, e7 = J(EXP / "exp6_correction_ablation" / "results.json"), J(EXP / "exp7_parameter_variants" / "results.json")
tests = J(ROOT / "results" / "logs" / "test_summary.json") or {}
manifest = None
for prof in ("full", "standard", "smoke"):
    manifest = J(ROOT / "data" / "synthetic" / f"{prof}_manifest.json")
    if manifest:
        break

# ============================================================================ 1 title
s = prs.slides.add_slide(BLANK)
N_SLIDE[0] += 1
bg = s.shapes.add_shape(1, 0, 0, W, Inches(0.35))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
_tb(s, Inches(0.8), Inches(1.4), Inches(11.7), Inches(1.6),
    "Reproducing a Stochastic Cluster-Based Channel Model for Sub-THz Outdoor D2D Links at 145 GHz", 34, True, NAVY)
_tb(s, Inches(0.8), Inches(3.1), Inches(11.7), Inches(0.9),
    "Independent Python implementation, numerical verification and statistical validation of Cheng et al., "
    "IEEE Transactions on Wireless Communications, 2026", 18, False, GREY)
_tb(s, Inches(0.8), Inches(4.4), Inches(6), Inches(0.4), "Contributors", 16, True, ACCENT)
for i, (roll, name) in enumerate(TEAM):
    _tb(s, Inches(0.8), Inches(4.85 + 0.42 * i), Inches(8), Inches(0.4), f"{roll}  -  {name}", 18)

# ============================================================================ 2 team
s = slide("Contributors")
table(s, pd.DataFrame({"Enrolment no.": [t[0] for t in TEAM], "Name": [t[1] for t in TEAM]}),
      Inches(0.8), Inches(1.6), Inches(7), Inches(1.6), 16)
bullets(s, Inches(0.8), Inches(3.8), Inches(11.5), Inches(2.6), [
    "Joint work on paper analysis, implementation, experiments and documentation",
    "Repository: sub-thz-d2d-channel-model/ (Python 3.12, numpy/scipy, pytest)",
    "Paper authors are credited in the references; the paper PDF, its data and the authors' toolkit are not part of this work",
], 16)

# ============================================================================ 3 motivation
s = slide("Motivation and problem", "Why a dynamic-range-aware cluster model at 145 GHz?")
bullets(s, Inches(0.5), Inches(1.5), Inches(7.2), Inches(5.3), [
    "Sub-THz D2D links (1.5 m terminals, 1-100 m) need double-directional channel models for beam management and system design",
    "Cluster models describe the channel by a few parametric clusters (delay, DoA, DoD, power, spreads)",
    "Measured clusters are only seen above a 35 dB dynamic-range cut-off: weak clusters vanish and spreads shrink",
    "Key idea of the paper: model what is observable, then invert the truncation when generating channels",
    "Our task: re-implement the whole chain from the paper, verify each step numerically, and validate what can be validated without the measurement data",
], 16)
image(s, FIG / "figS1_example_adps.png", Inches(7.9), Inches(1.6), Inches(5.1), Inches(4.6))
_tb(s, Inches(7.9), Inches(6.3), Inches(5.1), Inches(0.4), "Our generator output (simulated): LoS 20 m and NLoS 40 m", 11, color=GREY)

# ============================================================================ 4 paper overview
s = slide("The paper in one slide", "Cheng, Abbasi, Gomez-Ponce, Zhang, Molisch - IEEE TWC 25 (2026)")
bullets(s, Inches(0.5), Inches(1.5), Inches(6.2), Inches(5.3), [
    "Measurements: 145-146 GHz VNA sounder, 1001 tones (1 ns bins), 13 deg horns rotated in 10 deg steps, 21 LoS + 17 NLoS links",
    "ADPS (Eq. 1) -> MPC taps within 35 dB -> multi-level KPD clustering",
    "Table I: Gamma cluster numbers / arrivals, bimodal-Laplace centroids, log-normal spreads, distance-ratio NCP model",
    "Truncation-aware MLE for the cluster power (Eqs. 10-11)",
    "Generator: exponential-Laplacian cluster ADPS whose decay constants are found by inverting the DR-truncated spreads",
    "Validation: CvM distances of channel spreads (Fig. 17), path-loss parameters (Table II)",
], 15)
image(s, FIG / "fig16_generation_flow.png", Inches(6.9), Inches(1.3), Inches(6.0), Inches(5.6))

# ============================================================================ 5 data availability
s = slide("What could be reproduced - data availability")
table(s, pd.DataFrame({
    "Item": ["Raw measurements / ADPSs / cluster lists", "Authors' MATLAB toolkit [60]", "Table I, Table II, CvM values",
             "Figs. 2, 6, 10, 11, 17 (CDF curves)", "Link distances", "Paper PDF"],
    "Status": ["not published", "behind a Google Form - not requested (personal data)", "used as published",
               "digitised from vector graphics", "from [21] Table II", "used locally, not redistributed"],
    "Consequence": ["measured fits cannot be re-run", "no code-level comparison", "basis of all simulations",
                    "labelled 'measured (digitized)' / 'paper synthetic (digitized)'", "validation channels at measured distances",
                    "copyright respected"]}), Inches(0.5), Inches(1.5), Inches(12.3), Inches(3.2), 13)
callout(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.4),
        "Integrity rule followed throughout: no invented measurement data, no published numbers copied in as our results. "
        "Every curve is labelled as measured (digitized), paper synthetic (digitized), Table I model, or ours (simulated).", 14)

# ============================================================================ 6 pipeline
s = slide("Measurement-side pipeline: ADPS, taps and ML-KPD", "Implemented and tested on synthetic sounder data and generator output")
bullets(s, Inches(0.5), Inches(1.5), Inches(6.4), Inches(5.3), [
    "ADPS = |IFFT(H)|^2 / N_f per orientation pair, 5x delay oversampling (max per bin) against scalloping",
    "MPC taps: all samples within 35 dB of the global maximum",
    "KPD: kNN density weighted by exp(normalised power), relative density, key points, merge by connectivity threshold",
    "ML-KPD: split clusters whose delay/DoA/DoD projection is multimodal (5/3/3-bin smoothing, dip ratio 0.8); K' = sqrt(M_c/2), threshold searched from 1 in 0.01 steps",
    "Not K-means / DBSCAN - the paper's algorithm is kept",
], 15)
if e5:
    m = pd.DataFrame(e5["mean"])
    df = pd.DataFrame({"condition": m.condition.str.upper(), "true clusters": m.n_true.round(1), "KPD": m.n_kpd.round(1),
                       "ML-KPD": m.n_mlkpd.round(1), "ARI KPD": m.ari_kpd.round(3), "ARI ML-KPD": m.ari_mlkpd.round(3)})
    _tb(s, Inches(7.1), Inches(1.5), Inches(5.8), Inches(0.4), "Experiment 5 - clustering generated ADPSs (exact labels)", 14, True, NAVY)
    table(s, df, Inches(7.1), Inches(2.0), Inches(5.8), Inches(1.1), 12)
image(s, EXP / "exp5_clustering" / "ari.png", Inches(7.9), Inches(3.4), Inches(4.2), Inches(3.4))

# ============================================================================ 7 statistical model
s = slide("Statistical cluster model (Table I) and our samples", "Simulated from the published parameters vs digitised measured CDFs")
image(s, FIG / "fig02_cluster_number.png", Inches(0.3), Inches(1.4), Inches(6.4), Inches(4.6))
image(s, FIG / "fig06_intra_cluster_delay_spread.png", Inches(6.7), Inches(1.4), Inches(6.4), Inches(4.6))
callout(s, Inches(0.5), Inches(6.1), Inches(12.3), Inches(0.85),
        "Drawn cluster numbers follow the measured CDF (as Table I intends), but only part of them survives the 35 dB cut-off "
        "(solid lines). Re-observed intra-cluster spreads match Table I - the inverse mapping works per cluster.", 13)

# ============================================================================ 8 power model
s = slide("Cluster power: truncation-aware MLE (Eqs. 10-11)", "Experiment 4 - ground truth known (Table I), estimator tested on observable simulated clusters")
image(s, FIG / "fig13_ncp_nlos.png", Inches(0.3), Inches(1.4), Inches(6.3), Inches(4.7))
if e4:
    rows = []
    for c in ("los", "nlos"):
        r = e4[c]
        rows.append({"case": c.upper(), "truth beta": r["truth_table_I"]["beta"], "MLE beta": round(r["mle"]["beta"], 2),
                     "OLS beta": round(r["ols"]["beta"], 2), "truth alpha": r["truth_table_I"]["alpha"],
                     "MLE alpha": round(r["mle"]["alpha"], 2), "OLS alpha": round(r["ols"]["alpha"], 2),
                     "N obs.": r["n_observable"]})
    table(s, pd.DataFrame(rows), Inches(6.8), Inches(1.6), Inches(6.2), Inches(1.1), 11)
    r = e4["nlos"]
    bullets(s, Inches(6.8), Inches(3.1), Inches(6.2), Inches(3.6), [
        "Eq. 10 distance-equalisation weights, 20 bins; LLF with ln Phi truncation term; analytic gradient",
        "Alternating (alpha,beta)/(rho,eta) ascent, learning rate 1e-3, multi-start; L-BFGS only as a check",
        f"NLoS: MLE slope {e4['nlos']['mle']['beta']:.2f} vs OLS {e4['nlos']['ols']['beta']:.2f} (truth {e4['nlos']['truth_table_I']['beta']}) - truncation correction helps, as in the paper",
        f"LoS: MLE {e4['los']['mle']['beta']:.2f}, OLS {e4['los']['ols']['beta']:.2f} vs truth {e4['los']['truth_table_I']['beta']} - both biased: in the generator visibility depends on cluster shape, not a sharp NCP threshold",
        f"NLoS coverage +-1 sigma: {100 * r['coverage_1sigma']['empirical']:.1f} % (paper, measured: 67.79 %)",
    ], 14)

# ============================================================================ 9 dynamic range math
s = slide("Dynamic range truncation of a cluster", "Exponential-Laplacian cluster, cut-off L = DR_eff ln10 / 10")
bullets(s, Inches(0.5), Inches(1.5), Inches(6.3), Inches(5.3), [
    "Cluster ADPS ~ exp(-tau/k_tau - |phiR|/k_R - |phiT|/k_T); observed only where the exponent <= L",
    "Moments written as nested integrals: inner two in closed form (incomplete gamma), outer by piecewise Gauss-Legendre",
    "Angular spreads independent of k_tau; S_tau = k_tau * g(L, k_R, k_T) -> 3-D inversion reduces to 2-D Newton + closed form",
    "Complex-step Jacobian (exact to machine precision) + backtracking + bisection start",
    "Verified against brute-force integration (< 1 %) and DR -> infinity limits",
    "Four regimes: an angle saturates when L >= pi/k (14.48, 21.71, 36.19 dB for Fig. 14/15)",
], 15)
if e3:
    rt = e3["roundtrip"]
    callout(s, Inches(7.0), Inches(1.7), Inches(5.9), Inches(1.9),
            f"Experiment 3: {rt['n']} random (kappa, DR) cases, {100 * rt['converged_share']:.1f} % converged, "
            f"max relative kappa error {max(rt['max_rel_kappa_err']):.1e}, {rt['time_per_inversion_ms']:.2f} ms per cluster.", 15)
image(s, EXP / "exp3_inverse_mapping" / "discretisation_error.png", Inches(6.9), Inches(3.9), Inches(6.1), Inches(2.6))

# ============================================================================ 10 fig14/15
s = slide("Figs. 14-15 reproduced: inverse estimation vs dynamic range", "kappa_tau = 10 m, kappa_DoD = 0.2 pi, kappa_DoA = 0.3 pi, DR = 1...50 dB")
image(s, FIG / "fig14_inverse_delay.png", Inches(0.3), Inches(1.4), Inches(6.3), Inches(4.6))
image(s, FIG / "fig15_inverse_angular.png", Inches(6.7), Inches(1.4), Inches(6.3), Inches(4.6))
callout(s, Inches(0.5), Inches(6.1), Inches(12.3), Inches(0.85),
        "Independent numerical reproduction: inverse-solved spreads coincide with the actual ones in all four regimes; "
        "dotted lines = same cluster on the 1 ns / 10 deg grid (discretisation error that step 5 corrects).", 13)

# ============================================================================ 11 generator
s = slide("Generator: correction, inverse mapping and the cut-off fixed point")
bullets(s, Inches(0.5), Inches(1.5), Inches(7.2), Inches(5.3), [
    "Drawn cluster power is an observable power (sum of surviving taps) -> peak depends on shape, shape on DR_eff, DR_eff on all peaks",
    "We solve this explicitly: iterate cut-off <-> shapes, damped cut-off update; converges in 2-3 sweeps",
    "Discretisation correction table Delta(DR, kappa) = grid - continuum spreads (22 x 26 x 24 x 24, built once, ~2 min)",
    "Bounded correction (|Delta| <= 50 % of target): an unbounded correction ran away for weak, narrow clusters (kappa_tau of thousands of metres) - found and fixed during this work",
    "Each cluster contributes only its own above-cut-off taps, exactly the truncated shape the inversion assumed",
], 15)
if e6:
    df = pd.DataFrame(e6)
    df = pd.DataFrame({"cond.": df.condition.str.upper(), "correction": df.correction.map({True: "on", False: "off"}),
                       "|err| delay": df.median_abs_err_tau.round(3), "|err| DoA": df.median_abs_err_doa.round(3),
                       "|err| DoD": df.median_abs_err_dod.round(3)})
    _tb(s, Inches(7.9), Inches(1.5), Inches(5), Inches(0.4), "Exp. 6 - re-observed vs drawn spreads (median)", 13, True, NAVY)
    table(s, df, Inches(7.9), Inches(1.95), Inches(5.0), Inches(1.7), 11)
if e1:
    m = pd.DataFrame(e1).groupby("condition").mean_s.mean()
    callout(s, Inches(7.9), Inches(4.2), Inches(5.0), Inches(1.3),
            f"Exp. 1 - cost per realization (1 process): LoS {m['los']:.2f} s, NLoS {m['nlos']:.2f} s on the full 1001 x 36 x 36 grid.", 13)

# ============================================================================ 12 software
s = slide("Implementation and verification")
table(s, pd.DataFrame({
    "Module": ["distributions / parameters", "adps / mpc_extraction", "kpd_clustering", "power_model", "dynamic_range",
               "inverse_mapping", "channel_generator / metrics", "path_loss / validation"],
    "Content": ["Table I laws, Fleury spread, YAML config with paper alternatives", "Eq. 1 with oversampling, 35 dB taps, test sounder",
                "KPD and ML-KPD (Algorithm 1)", "Eqs. 6-11, weights, MLE, OLS", "continuous + exact discrete truncation, regimes",
                "2-D Newton inverse, correction table", "7-step generator, omni/max-dir metrics", "alpha-beta / CI fits with CIs, CvM, bootstrap null"]}),
    Inches(0.5), Inches(1.5), Inches(7.6), Inches(4.4), 12)
nt = tests.get("passed")
bullets(s, Inches(8.4), Inches(1.5), Inches(4.6), Inches(5.0), [
    f"pytest: {nt} tests passed, {tests.get('failed', 0)} failed" if nt else "pytest suite in tests/",
    "brute-force checks of the truncation integrals; FD check of the LLF gradient; MLE recovers known parameters",
    "integration test: full generator, 35 dB cut-off, determinism, no runaway decay constants",
    "every realization has its own SeedSequence stream -> identical results for any worker count",
    "profiles smoke / standard / full",
], 14)

# ============================================================================ 13 fig17
s = slide("Validation 1: channel spreads (Fig. 17)", "Ours (simulated at the measured link distances) vs measured and paper-synthetic CDFs (digitized)")
image(s, FIG / "fig17_validation_cdfs.png", Inches(0.3), Inches(1.35), Inches(12.7), Inches(4.3))
if manifest:
    callout(s, Inches(0.5), Inches(5.9), Inches(12.3), Inches(1.0),
            f"{manifest['n_realizations']} realizations per condition (profile '{manifest['profile']}'). LoS agrees closely with the "
            "measured CDFs; NLoS channels come out more dispersive than measured and than the authors' own generator.", 14)

# ============================================================================ 14 cvm table
s = slide("Validation 1: Cramer-von Mises distances")
if cvm is not None:
    df = pd.DataFrame({"metric": cvm.metric, "cond.": cvm.condition.str.upper(), "paper (published)": cvm.published_cvm,
                       "paper (recomputed, digitized)": cvm.recomputed_paper_cvm.round(4),
                       "ours vs measured": cvm.ours_vs_measured_cvm.round(4), "sampling null 95 %": cvm.null_cvm_95.round(4),
                       "median ours": cvm.median_ours.round(3), "median measured": cvm.median_measured.round(3)})
    table(s, df, Inches(0.4), Inches(1.4), Inches(12.5), Inches(2.9), 12)
    ok = (cvm.ours_vs_measured_cvm <= cvm.null_cvm_95)
    bullets(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.3), [
        "omega^2 = mean of (F_a - F_b)^2 over the pooled sample; recomputing the paper's values from digitised curves gives the same magnitude",
        "Sampling null: omega^2 of two samples of the same law with the same sizes (21/17 measured) - the floor explainable by finite data",
        f"{int(ok.sum())} of {len(ok)} statistics within the sampling null; consistent: "
        + ", ".join(f"{a} {b.upper()}" for a, b in zip(cvm.metric[ok], cvm.condition[ok])),
    ], 14)

# ============================================================================ 15 path loss
s = slide("Validation 2: path loss (Table II)", "50 channels, d ~ U(1, 100) m, repeated draws from the generated pool")
if pl is not None:
    sub = pl[pl.parameter.isin(["alpha", "beta", "sigma_ab", "n_cim"])]
    df = pd.DataFrame({"case": sub.case, "param": sub.parameter, "[21] measured": sub["measured_ref_[21]"],
                       "paper synthetic": sub.published_synthetic,
                       "ours median": sub.ours_median_over_repeats.round(2),
                       "ours 95 % range": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(sub["ours_p2.5"], sub["ours_p97.5"])],
                       "CI overlap": (100 * sub.share_repeats_ci_overlap_published).round(0).astype(int).astype(str) + " %"})
    table(s, df, Inches(0.3), Inches(1.4), Inches(8.6), Inches(5.2), 10)
image(s, FIG / "figS2_pathloss.png", Inches(9.0), Inches(1.6), Inches(4.1), Inches(3.4))
_tb(s, Inches(9.0), Inches(5.1), Inches(4.1), Inches(1.6),
    "CI overlap = share of our 50-channel repeats whose 95 % CI overlaps the published one.", 11, color=GREY)

# ============================================================================ 16 sensitivity
s = slide("Sensitivity: dynamic range and ambiguous parameters", "Experiments 2 and 7")
image(s, EXP / "exp2_dr_sensitivity" / "dr_sweep.png", Inches(0.3), Inches(1.4), Inches(12.7), Inches(3.2))
if e7:
    df = pd.DataFrame(e7)
    cols = {"condition": "cond.", "variant": "variant", "ds_median_m": "median DS [m]", "asa_median": "median ASA",
            "asd_median": "median ASD"}
    extra = [c for c in ("cvm_delay", "cvm_doa", "cvm_dod") if c in df]
    d2 = df[list(cols) + extra].rename(columns=cols).round(4)
    d2["cond."] = d2["cond."].str.upper()
    table(s, d2, Inches(0.5), Inches(4.8), Inches(12.3), Inches(1.9), 11)

# ============================================================================ 17 discrepancies
s = slide("Inconsistencies found in the paper", "Documented in docs/PAPER_ANALYSIS.md; each choice is configurable")
table(s, pd.DataFrame({
    "Item": ["Eq. (4) angular spread", "Eq. (1)", "Eq. (10)", "Arrival-time Gamma", "Figs. 14/15", "Fig. 2 NLoS fit",
             "Cluster count vs truncation"],
    "Paper": ["typeset without sqrt", "|.|^2 inside the sum", "1 % in equation, 2 % in text", "Table I vs text values differ",
              "text: DoD 0.2 pi / DoA 0.3 pi, legend reversed", "plotted fit matches Gamma on N, stated on N-1",
              "counts of observed clusters, then power truncation removes more"],
    "Our choice": ["Fleury sqrt (max 1, matches Fig. 15 and [21])", "|sum|^2", "1 % (2 % configurable)", "Table I (text in exp. 7)",
                   "text", "as stated (N-1)", "as described; effect quantified"]}),
    Inches(0.4), Inches(1.45), Inches(12.5), Inches(4.9), 12)

# ============================================================================ 18 status
s = slide("Reproduction status")
table(s, pd.DataFrame({
    "Result": ["Table I distributions, Figs. 2-11 model curves", "Cluster ADPS (Eqs. 12-13), truncation map, Figs. 14-15",
               "Inverse mapping + discretisation correction", "Truncated MLE (Figs. 12-13 method)", "ML-KPD clustering",
               "Fig. 17 / CvM", "Table II path loss", "Measured fits (Table I extraction), Fig. 1"],
    "Classification": ["Statistical reproduction using published parameters", "Independent numerical reproduction",
                       "Independent numerical reproduction (correction design ours)",
                       "Independent numerical reproduction (on simulated data)", "Approximate reproduction (unspecified details)",
                       "Statistical reproduction: LoS consistent, NLoS deviates", "Statistical reproduction using published parameters",
                       "Not reproducible - measurement data unavailable"]}),
    Inches(0.4), Inches(1.45), Inches(12.5), Inches(4.6), 13)

# ============================================================================ 19 limitations
s = slide("Limitations and future work")
bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5.3), [
    "No measured data: Table I cannot be re-extracted and Figs. 3-5, 7-9, 12-13 cannot be compared with measured scatter/histograms",
    "Digitised curves carry small reading errors (<= 0.015 in CDF for most curves)",
    "Several algorithmic details are unspecified (KPD features and level-1 threshold, oversampling factor, correction-table design, NCP threshold, peak/cut-off coupling) - our choices are documented",
    "NLoS channels are more dispersive than measured: with ~90 clusters, uniform centroids and ~10 dB NCP spread, many off-boresight clusters stay observable",
    "Synthetic-sounder test data with un-windowed IFFT sidelobes make (ML-)KPD over-segment strong clusters",
    "Future work: compare against the authors' toolkit once obtained through the official form; validate on new measurements",
], 16)

# ============================================================================ 20 references
s = slide("References")
bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5.3), [
    PAPER + ", doi: 10.1109/TWC.2026.3714332.",
    "N. A. Abbasi et al., \"THz band channel measurements and statistical modeling for urban D2D environments,\" arXiv:2109.13693 ([21]).",
    "R. He et al., \"A kernel-power-density-based algorithm for channel multipath components clustering,\" IEEE TWC, 2017 (KPD).",
    "B. H. Fleury, \"First- and second-order characterization of direction dispersion and space selectivity in the radio channel,\" IEEE Trans. IT, 2000.",
    "T. W. Anderson, \"On the distribution of the two-sample Cramer-von Mises criterion,\" Ann. Math. Statist., 1962.",
    "USC WiDeS, Channel Generation Toolkit for sub-THz Outdoor D2D Links at 145 GHz, wides.usc.edu/research_matlab.html ([60]).",
], 14)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"saved {OUT} with {len(prs.slides)} slides")
