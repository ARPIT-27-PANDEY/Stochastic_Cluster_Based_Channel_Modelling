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
