"""Run the Qt channel emulator off-screen for a few seconds and save screenshots.

    <radioconda>/python gnuradio/screenshot_emulator.py

Writes results/gnuradio/emulator_<condition>_<set>.png (used on the slides).
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("SUBTHZ_REPO", str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(HERE)

from PyQt5 import Qt  # noqa: E402

app = Qt.QApplication(sys.argv)
import subthz_channel_emulator as m  # noqa: E402

OUT = ROOT / "results" / "gnuradio"
OUT.mkdir(parents=True, exist_ok=True)
cases = [("los", 20.0, 0, "", "table1"), ("nlos", 40.0, 0, "", "table1")]
if (ROOT / "config" / "channel_parameters_calibrated.yaml").exists():
    cases += [("nlos", 40.0, 0, "config/channel_parameters_calibrated.yaml", "calibrated")]
for cond, d, idx, model, tag in cases:
    tb = m.subthz_channel_emulator()
    tb.set_condition(cond)
    tb.set_d_m(d)
    tb._d_m_win.d_widget.setValue(d)        # move the GUI widget too (the setter only changes the variable)
    tb.set_index(idx)
    tb._index_win.d_widget.setValue(idx)
    tb.set_model(model)
    tb.resize(1600, 1120)
    tb.show()
    tb.start()
    tb.flowgraph_started.set()
    t0 = time.time()
    # wait until the channel is drawn and shown (generation takes ~1-3 s per channel)
    while time.time() - t0 < 90.0 and (time.time() - t0 < 8.0 or str(tb.get_stats()).startswith("drawing")):
        app.processEvents()
        time.sleep(0.05)
    t1 = time.time()
    while time.time() - t1 < 4.0:        # let the displays refresh with the new channel
        app.processEvents()
        time.sleep(0.05)
    path = OUT / f"emulator_{cond}_{tag}.png"
    tb.grab().save(str(path))
    print("saved", path, "|", tb.get_stats())
    tb.stop()
    tb.wait()
    tb.hide()
