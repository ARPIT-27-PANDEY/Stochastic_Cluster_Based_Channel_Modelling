"""Read the published curves back from the vector figures of the paper PDF.

Usage:
    python scripts/digitize_published_figures.py --pdf path/to/paper.pdf

Writes data/processed/digitized/*.csv and a manifest.json.  The axis calibration was
derived from the gridline positions in the PDF (see docs/DATA_AVAILABILITY.md).
These values are DIGITIZED FROM THE PUBLISHED FIGURES, not raw measurement data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from channel_model.digitize import Panel, extract_curves, step_cdf_to_samples  # noqa: E402

OUT = ROOT / "data" / "processed" / "digitized"

# (panel, ordered roles of the main curves as drawn, sample sizes of step CDFs; None = infer from jump height)
PANELS = [
    (Panel("fig02_cluster_number", 5, (340, 40, 530, 190), (1, 300), (0, 1), True),
     ["los_measured", "los_gamma_fit", "nlos_measured", "nlos_gamma_fit"], {"los_measured": 21, "nlos_measured": 17}),
    (Panel("fig06_delay_spread", 7, (90, 40, 270, 190), (0.05, 5), (0, 1), True),
     ["loscluster_measured", "loscluster_fit", "los_measured", "los_fit", "nlos_measured", "nlos_fit"], {}),
    (Panel("fig10_doa_spread", 8, (350, 40, 530, 190), (0.05, 0.6), (0, 1), True),
     ["loscluster_measured", "loscluster_fit", "los_measured", "los_fit", "nlos_measured", "nlos_fit"], {}),
    (Panel("fig11_dod_spread", 8, (350, 215, 530, 370), (0.05, 0.6), (0, 1), True),
     ["loscluster_measured", "loscluster_fit", "los_measured", "los_fit", "nlos_measured", "nlos_fit"], {}),
    (Panel("fig17a_delay_spread", 15, (40, 45, 215, 180), (0, 40), (0, 1)),
     ["los_measured", "los_paper_synthetic", "nlos_measured", "nlos_paper_synthetic"],
     {"los_measured": 21, "nlos_measured": 17, "los_paper_synthetic": None, "nlos_paper_synthetic": None}),
    (Panel("fig17b_doa_spread", 15, (215, 45, 390, 180), (0, 1), (0, 1)),
     ["los_measured", "los_paper_synthetic", "nlos_measured", "nlos_paper_synthetic"],
     {"los_measured": 21, "nlos_measured": 17, "los_paper_synthetic": None, "nlos_paper_synthetic": None}),
    (Panel("fig17c_dod_spread", 15, (390, 45, 580, 180), (0, 1), (0, 1)),
     ["los_measured", "los_paper_synthetic", "nlos_measured", "nlos_paper_synthetic"],
     {"los_measured": 21, "nlos_measured": 17, "los_paper_synthetic": None, "nlos_paper_synthetic": None}),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, help="local copy of the IEEE TWC paper")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "Cheng et al., IEEE TWC vol. 25, 2026 - vector figures",
                "note": "digitized from the published figures; not raw measurement data", "curves": []}
    for panel, roles, nsamp in PANELS:
        curves = [c for c in extract_curves(args.pdf, panel)
                  if c["n_items"] >= 25 and (c["y"].max() - c["y"].min()) > 0.9]
        if len(curves) != len(roles):
            raise RuntimeError(f"{panel.name}: found {len(curves)} main curves, expected {len(roles)}")
        for role, c in zip(roles, curves):
            fn = OUT / f"{panel.name}__{role}.csv"
            np.savetxt(fn, np.column_stack([c["x"], c["y"]]), delimiter=",", header="x,y", comments="", fmt="%.6g")
            entry = {"figure": panel.name, "role": role, "file": fn.name, "page": panel.page,
                     "x_range": panel.x_range, "x_log": panel.x_log, "color": c["color"], "n_points": int(len(c["x"]))}
            if role in nsamp:
                s, n_clip = step_cdf_to_samples(c["x"], c["y"], nsamp[role], x_max=panel.x_range[1])
                entry["n_clipped_at_axis_max"] = n_clip
                sf = OUT / f"{panel.name}__{role}__samples.csv"
                np.savetxt(sf, s, delimiter=",", header="value", comments="", fmt="%.6g")
                entry["samples_file"] = sf.name
                entry["n_samples"] = int(s.size)
            manifest["curves"].append(entry)
            print(f"{panel.name:24s} {role:22s} points={len(c['x']):5d}" + (f"  samples={entry.get('n_samples')}" if role in nsamp else ""))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
