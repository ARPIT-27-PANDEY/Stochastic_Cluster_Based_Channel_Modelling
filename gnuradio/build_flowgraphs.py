"""Write the GNU Radio Companion flowgraphs of this project (GNU Radio 3.10).

    <radioconda>/python gnuradio/build_flowgraphs.py

Produces
  gnuradio/subthz_sounder_batch.grc      no-GUI VNA channel sounder (validation runs, Fig. 17 / Table II)
  gnuradio/subthz_channel_emulator.grc   Qt GUI channel emulator (PDP, APS, wideband signal through the channel)

The .grc files are plain YAML; this script only exists so that every block parameter is
written consistently.  Open the generated files in GNU Radio Companion as usual.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


class Str(str):
    pass


yaml.add_representer(Str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s,
                                                          style="|" if "\n" in s else None))


class FG:
    def __init__(self, fid, title, description, mode="qt_gui", window=(1500, 1000)):
        self.fid, self.title, self.desc, self.mode, self.window = fid, title, description, mode, window
        self.blocks, self.conns = [], []

    def add(self, _name, _bid, _xy=(8, 8), **params):
        self.blocks.append({"name": _name, "id": _bid,
                            "parameters": {k: Str(str(v)) for k, v in params.items()},
                            "states": {"bus_sink": False, "bus_source": False, "bus_structure": None,
                                       "coordinate": [float(_xy[0]), float(_xy[1])], "rotation": 0,
                                       "state": "enabled"}})
        return _name

    def link(self, a, b, ap=0, bp=0):
        self.conns.append([a, str(ap), b, str(bp)])

    def var(self, name, value, comment="", _xy=(8, 8)):
        return self.add(name, "variable", _xy, value=value, comment=comment)

    def param(self, name, label, typ, value, _xy=(8, 8)):
        return self.add(name, "parameter", _xy, label=label, type=typ, value=value, short_id="", hide="none")

    def epy(self, name, source, _xy=(8, 8), **params):
        return self.add(name, "epy_block", _xy, _source_code=textwrap.dedent(source).lstrip(), **params)

    def save(self):
        doc = {"options": {"parameters": {
            "author": "Group 10 - P. Bhaskar, A. Tripathi, A. K. Pandey", "catch_exceptions": "True",
            "category": "[GRC Hier Blocks]", "cmake_opt": "", "comment": "", "copyright": "",
            "description": self.desc, "gen_cmake": "On", "gen_linking": "dynamic", "generate_options": self.mode,
            "hier_block_src_path": ".:", "id": self.fid, "max_nouts": "0", "output_language": "python",
            "placement": "(0,0)", "qt_qss_theme": "", "realtime_scheduling": "", "run": "True",
            "run_command": "{python} -u {filename}", "run_options": "run" if self.mode == "no_gui" else "prompt",
            "sizing_mode": "fixed", "thread_safe_setters": "", "title": self.title,
            "window_size": f"({self.window[0]},{self.window[1]})"},
            "states": {"bus_sink": False, "bus_source": False, "bus_structure": None, "coordinate": [8, 8],
                       "rotation": 0, "state": "enabled"}},
            "blocks": self.blocks, "connections": self.conns,
            "metadata": {"file_format": 1, "grc_version": "3.10.12.0"}}
        path = HERE / f"{self.fid}.grc"
        text = yaml.dump(doc, sort_keys=False, default_flow_style=False, width=4096, allow_unicode=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote", path)


# The embedded blocks import the project's model from <repo>/src.  GRC writes the block code
# next to the .grc file (<repo>/gnuradio), so the repository is one level up.
FIND_REPO = '''
def _repo_src():
    import os, sys
    cands = [os.environ.get("SUBTHZ_REPO", "")]
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    except NameError:
        pass
    cwd = os.getcwd()
    cands += [cwd, os.path.join(cwd, ".."), os.path.join(cwd, "..", "..")]
    for c in cands:
        src = os.path.join(c, "src")
        if c and os.path.isdir(os.path.join(src, "channel_model")):
            if src not in sys.path:
                sys.path.insert(0, os.path.abspath(src))
            return os.path.abspath(c)
    raise RuntimeError("cannot find the sub-thz-d2d-channel-model repository; set SUBTHZ_REPO")
'''

# =============================================================================
# 1. Batch VNA channel sounder
# =============================================================================
SOUNDER_SRC = FIND_REPO + '''

import json
import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):
    """Cluster-model channel + VNA sweep generator.

    For every channel realization (Sec. VI generator) and every one of the 36 x 36 Tx/Rx horn
    orientations this block emits one vector of 1001 complex tones H(f), 145-146 GHz.
    The first vector of a realization carries tags: rid, d_m, gain_db (peak normalisation)
    and truth (metrics of the generator's own ADPS, for verification of the sounder chain).
    The flowgraph ends after n_channels realizations.
    """

    def __init__(self, condition="los", n_channels=4, start_index=0, seed=20260723,
                 distances="measured", params_file=""):
        gr.sync_block.__init__(self, name="Sub-THz cluster channel + VNA sweep",
                               in_sig=None, out_sig=[(np.complex64, 1001)])
        self.condition, self.n_channels, self.start_index = condition, int(n_channels), int(start_index)
        self.seed, self.distances, self.params_file = int(seed), distances, params_file
        self._k = 0              # realization counter
        self._buf = None         # sweeps of the current realization
        self._pos = 0
        self._gs = None

    def _next(self):
        if self._gs is None:
            _repo_src()
            from channel_model import gr_support
            self._gs = gr_support
        gs = self._gs
        i = self.start_index + self._k
        d = gs.link_distance(self.condition, self.distances, self.seed, i)
        real = gs.realization(self.condition, d, self.seed, i, self.params_file)
        H, gain = gs.sounder_sweeps(real.adps, np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(7, i))))
        off = self.nitems_written(0)
        for key, val in (("rid", pmt.from_long(i)), ("d_m", pmt.from_double(d)), ("gain_db", pmt.from_double(gain)),
                         ("truth", pmt.intern(gs.truth_tag(real)))):
            self.add_item_tag(0, off, pmt.intern(key), val)
        gs.realization.cache_clear()
        self._buf, self._pos = H, 0
        self._k += 1

    def work(self, input_items, output_items):
        out = output_items[0]
        if self._buf is None or self._pos >= self._buf.shape[0]:
            if self._k >= self.n_channels:
                return -1                     # WORK_DONE: all realizations sounded
            self._next()
        n = min(len(out), self._buf.shape[0] - self._pos)
        out[:n] = self._buf[self._pos:self._pos + n]
        self._pos += n
        return n
'''

SINK_SRC = FIND_REPO + '''

import csv
import json
import os
import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):
    """ADPS assembly, 35 dB MPC threshold and channel metrics (Secs. IV-A, VIII).

    Input: one 1001-bin delay profile |IFFT(H)|^2 / N_f per orientation pair.
    Every 36 x 36 vectors form one ADPS; the metrics are appended to out_csv together with
    the generator's own (truth) metrics carried in the stream tags.
    """

    def __init__(self, out_csv="sounder.csv", condition="los", dr_db=35.0):
        gr.sync_block.__init__(self, name="ADPS + 35 dB taps + channel metrics",
                               in_sig=[(np.float32, 1001)], out_sig=None)
        self.out_csv, self.condition, self.dr_db = out_csv, condition, float(dr_db)
        self._rows, self._meta, self._acc, self._gs = [], None, [], None

    def _flush(self):
        if self._meta is None or not self._acc:
            return
        gs = self._gs
        stack = np.concatenate(self._acc, axis=0)
        if stack.shape[0] == 36 * 36:
            m = gs.metrics_from_sounding(stack, self._meta["gain_db"], dr_db=self.dr_db)
            row = {"rid": self._meta["rid"], "condition": self.condition, "d_m": self._meta["d_m"],
                   **{f"gr_{k}": v for k, v in m.items()},
                   **{f"truth_{k}": v for k, v in self._meta["truth"].items()}}
            self._rows.append(row)
        self._acc, self._meta = [], None

    def work(self, input_items, output_items):
        if self._gs is None:
            _repo_src()
            from channel_model import gr_support
            self._gs = gr_support
        x = input_items[0]
        n = len(x)
        start = self.nitems_read(0)
        tags = sorted(self.get_tags_in_window(0, 0, n, pmt.intern("rid")), key=lambda t: t.offset)
        cuts = [t.offset - start for t in tags] + [n]
        prev = 0
        for t, cut in zip([None] + tags, cuts):
            if t is not None:
                self._flush()
                meta = {"rid": pmt.to_long(t.value)}
                for tt in self.get_tags_in_range(0, t.offset, t.offset + 1):
                    k = pmt.symbol_to_string(tt.key)
                    if k in ("d_m", "gain_db"):
                        meta[k] = pmt.to_double(tt.value)
                    elif k == "truth":
                        meta["truth"] = json.loads(pmt.symbol_to_string(tt.value))
                self._meta = meta
            if cut > prev and self._meta is not None:
                self._acc.append(np.array(x[prev:cut]))
            prev = cut
        return n

    def stop(self):
        self._flush()
        if self._rows:
            new = not os.path.exists(self.out_csv)
            with open(self.out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self._rows[0]))
                if new:
                    w.writeheader()
                w.writerows(self._rows)
        print(f"[sounder sink] wrote {len(self._rows)} channel(s) to {self.out_csv}")
        return True
'''

fg = FG("subthz_sounder_batch", "Sub-THz 145 GHz D2D channel sounder (batch validation)",
        "GNU Radio implementation of the measurement chain of Cheng et al. (IEEE TWC 2026): cluster-model channel "
        "-> per-orientation 1001-tone VNA sweep -> sounder noise -> IFFT -> |.|^2/N_f (Eq. 1) -> ADPS, 35 dB MPC "
        "threshold, delay/angular spreads and path gains (Sec. VIII). Run from the command line: "
        "python subthz_sounder_batch.py --condition nlos --n-channels 17 --out-csv out.csv",
        mode="no_gui")
fg.add("import_np", "import", (8, 230), imports="import numpy as np")
fg.param("condition", "Condition (los / nlos)", "str", "los", (272, 8))
fg.param("n_channels", "Number of channel realizations", "intx", 4, (488, 8))
fg.param("start_index", "First realization index", "intx", 0, (704, 8))
fg.param("seed", "Seed", "intx", 20260723, (920, 8))
fg.param("distances", "Link distances (measured / uniform)", "str", "measured", (1104, 8))
fg.param("params_file", "Parameter YAML ('' = Table I)", "str", "", (1336, 8))
fg.param("snr_db", "Sounder SNR per tone (dB below strongest tap)", "eng_float", 60.0, (1104, 150))
fg.param("out_csv", "Output CSV", "str", "sounder.csv", (1336, 150))
fg.var("nf", 1001, "1001 tones, 145-146 GHz (1 MHz spacing, 1 ns delay bins)", (272, 150))
fg.var("dr_db", 35.0, "MPC threshold below the ADPS maximum (Sec. IV-A)", (448, 150))
fg.epy("vna", SOUNDER_SRC, (8, 300), condition="condition", n_channels="n_channels", start_index="start_index",
       seed="seed", distances="distances", params_file="params_file")
fg.add("noise", "analog_noise_source_x", (8, 540), type="complex", noise_type="analog.GR_GAUSSIAN",
       amp="10 ** (-snr_db / 20)", seed=1)
fg.add("noise_vec", "blocks_stream_to_vector", (300, 556), type="complex", num_items="nf", vlen=1)
fg.add("rx_add", "blocks_add_xx", (420, 340), type="complex", num_inputs=2, vlen="nf")
fg.add("ifft", "fft_vxx", (600, 310), type="complex", fft_size="nf", forward="False", window="[]", shift="False",
       nthreads=1)
fg.add("mag2", "blocks_complex_to_mag_squared", (880, 340), vlen="nf")
fg.add("norm", "blocks_multiply_const_vxx", (1130, 340), type="float", const="[1.0 / nf] * nf", vlen="nf")
fg.epy("adps_metrics", SINK_SRC, (1420, 310), out_csv="out_csv", condition="condition", dr_db="dr_db")
fg.link("vna", "rx_add", 0, 0)
fg.link("noise", "noise_vec")
fg.link("noise_vec", "rx_add", 0, 1)
fg.link("rx_add", "ifft")
fg.link("ifft", "mag2")
fg.link("mag2", "norm")
fg.link("norm", "adps_metrics")
fg.save()

# =============================================================================
# 2. Interactive Qt channel emulator
# =============================================================================
EMU_SRC = FIND_REPO + '''

import threading
import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Sub-THz cluster-channel emulator (Sec. VI generator).

    Outputs (one vector set per update): omni PDP [dB] (1001 x 1 ns), DoA APS and DoD APS
    [dB] (36 x 10 deg).  taps() returns the omni channel impulse response used by the FIR
    filter, summary() the channel metrics shown in the GUI.  A new realization is drawn
    whenever condition, distance, realization index or parameter set change.
    """

    def __init__(self, condition="los", d_m=20.0, index=0, seed=20260723, params_file=""):
        gr.sync_block.__init__(self, name="Sub-THz cluster channel emulator", in_sig=None,
                               out_sig=[(np.float32, 1001), (np.float32, 36), (np.float32, 36)])
        self.condition, self.d_m, self.index, self.seed, self.params_file = condition, d_m, index, seed, params_file
        self._key, self._views, self._real, self._gs = None, None, None, None
        self._lock = threading.Lock()   # work() and the GUI probes may ask concurrently

    def _update(self):
        key = (self.condition, round(float(self.d_m), 2), int(self.index), int(self.seed), self.params_file)
        with self._lock:
            if key == self._key:
                return
            if self._gs is None:
                _repo_src()
                from channel_model import gr_support
                self._gs = gr_support
            real = self._gs.realization(key[0], key[1], key[3], key[2], key[4])
            self._views = self._gs.emulator_views(real)
            self._real, self._key = real, key

    def taps(self):
        try:
            self._update()
            return list(self._views["taps"])
        except Exception as e:  # keep the GUI alive while parameters are being edited
            print("emulator:", e)
            return [1.0 + 0j]

    def summary(self):
        try:
            self._update()
            return self._gs.emulator_summary(self._real)
        except Exception as e:
            return f"generator error: {e}"

    def work(self, input_items, output_items):
        self._update()
        n = min(len(o) for o in output_items)
        for k, name in enumerate(("pdp_db", "aps_r_db", "aps_t_db")):
            output_items[k][:n] = self._views[name]
        return n
'''

fg = FG("subthz_channel_emulator", "Sub-THz 145 GHz D2D cluster channel emulator",
        "Interactive GNU Radio emulator of the 145 GHz outdoor D2D cluster model (Cheng et al., IEEE TWC 2026): "
        "draws double-directional channels from Table I, shows the omni PDP and the DoA/DoD angular power spectra, "
        "and passes a 1 GHz-wide test signal through the channel impulse response with the GNU Radio FIR filter.",
        window=(1600, 1100))
fg.add("import_np", "import", (8, 200), imports="import numpy as np")
fg.var("samp_rate", "1e9", "1 GHz sounding bandwidth = 1 ns delay resolution", (200, 8))
fg.var("seed", 20260723, "", (360, 8))
fg.add("condition", "variable_qtgui_chooser", (520, 8), label="Propagation condition", type="string", value="'los'",
       widget="radio_buttons", orient="Qt.QHBoxLayout", num_opts=2, option0="'los'", label0="LoS",
       option1="'nlos'", label1="NLoS", gui_hint="0,0,1,1")
fg.add("model", "variable_qtgui_chooser", (720, 8), label="Parameter set", type="string", value="''",
       widget="radio_buttons", orient="Qt.QHBoxLayout", num_opts=2, option0="''", label0="Table I (published)",
       option1="'config/channel_parameters_calibrated.yaml'", label1="Calibrated", gui_hint="0,1,1,1")
fg.add("d_m", "variable_qtgui_range", (920, 8), label="Tx-Rx distance d_LoS (m)", start=1, stop=100, step=0.5,
       value=20, widget="counter_slider", rangeType="float", orient="QtCore.Qt.Horizontal", min_len=200,
       gui_hint="1,0,1,1")
fg.add("index", "variable_qtgui_range", (1120, 8), label="Realization #", start=0, stop=999, step=1, value=0,
       widget="counter_slider", rangeType="int", orient="QtCore.Qt.Horizontal", min_len=200, gui_hint="1,1,1,1")
fg.epy("emu", EMU_SRC, (8, 260), condition="condition", d_m="d_m", index="index", seed="seed", params_file="model")
fg.add("taps_probe", "variable_function_probe", (520, 120), block_id="emu", function_name="taps", function_args="",
       poll_rate=2, value="[1.0]")
fg.add("stats_probe", "variable_function_probe", (720, 120), block_id="emu", function_name="summary",
       function_args="", poll_rate=2, value="'drawing channel...'")
fg.add("stats", "variable_qtgui_label", (920, 120), label="Channel", type="string", value="stats_probe",
       formatter="None", gui_hint="2,0,1,2")
fg.add("thr_vec", "blocks_throttle2", (360, 60), type="float", samples_per_second=10, ignoretag="True",
       limit="auto", maximum="0.1", vlen=1001)

vs_common = dict(grid="True", autoscale="True", average="1.0", update_time="0.2", showports="False",
                 legend="True", ref_level=0, x_units='""', y_units='""')
fg.add("pdp_sink", "qtgui_vector_sink_f", (620, 30), name='"Omni PDP (max over orientation pairs), 35 dB MPC range"',
       vlen=1001, x_start=0, x_step=1, x_axis_label='"Excess delay (ns)"', y_axis_label='"Power (dB)"',
       ymin=-200, ymax=-40, nconnections=1, label1='"omni PDP"', color1='"blue"', width1=1, alpha1=1.0,
       gui_hint="3,0,1,2", **vs_common)
fg.add("aps_sink", "qtgui_vector_sink_f", (620, 290), name='"Angular power spectra"', vlen=36, x_start=-180,
       x_step=10, x_axis_label='"Angle (deg)"', y_axis_label='"Power (dB)"', ymin=-200, ymax=-40, nconnections=2,
       label1='"DoA (Rx)"', color1='"red"', width1=2, alpha1=1.0, label2='"DoD (Tx)"', color2='"dark green"',
       width2=2, alpha2=1.0, gui_hint="4,0,1,1", **vs_common)

fg.add("tx_sig", "analog_noise_source_x", (8, 720), type="complex", noise_type="analog.GR_GAUSSIAN", amp=1, seed=3)
fg.add("thr_sig", "blocks_throttle2", (330, 736), type="complex", samples_per_second=2e6, ignoretag="True",
       limit="auto", maximum="0.1", vlen=1)
fg.add("chan_fir", "fir_filter_xxx", (600, 760), type="ccc", decim=1, taps="taps_probe", samp_delay=0)
fs = dict(type="complex", fftsize=4096, wintype="window.WIN_BLACKMAN_hARRIS", fc="145.5e9", bw="samp_rate",
          grid="True", autoscale="False", average="0.05", ymin=-60, ymax=-20, update_time="0.20", legend="True",
          axislabels="True", label="Relative Gain", units="dB", freqhalf="True", showports="False",
          ctrlpanel="False", norm_window="False", tr_mode="qtgui.TRIG_MODE_FREE", tr_level="0.0", tr_chan=0,
          tr_tag='""')
fg.add("spec", "qtgui_freq_sink_x", (900, 720), name='"1 GHz test signal: transmitted vs after the channel (145-146 GHz)"',
       nconnections=2, label1="Tx (white)", color1='"blue"', label2="Rx (through channel)", color2='"red"',
       gui_hint="4,1,1,1", **fs)
fg.link("emu", "thr_vec", 0, 0)
fg.link("thr_vec", "pdp_sink")
fg.link("emu", "aps_sink", 1, 0)
fg.link("emu", "aps_sink", 2, 1)
fg.link("tx_sig", "thr_sig")
fg.link("thr_sig", "chan_fir")
fg.link("thr_sig", "spec", 0, 0)
fg.link("chan_fir", "spec", 0, 1)
fg.save()
