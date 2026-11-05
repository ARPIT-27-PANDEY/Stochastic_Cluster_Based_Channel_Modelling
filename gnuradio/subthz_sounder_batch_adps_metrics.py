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
