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
