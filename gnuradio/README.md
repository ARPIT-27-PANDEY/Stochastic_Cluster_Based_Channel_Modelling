# GNU Radio implementation

GNU Radio 3.10 flowgraphs (tested with GNU Radio 3.10.12 from radioconda on Windows 11).
The channel-model maths lives in `../src/channel_model`; the embedded Python blocks of both
flowgraphs import it (they look for the repository next to the `.grc` file, or in `$SUBTHZ_REPO`).

| File | What it is |
|---|---|
| `subthz_sounder_batch.grc` | No-GUI VNA channel sounder: cluster-model channel -> 1001-tone sweep for each of the 36 x 36 horn orientations -> sounder noise -> inverse FFT -> \|.\|^2 / N_f (Eq. 1) -> ADPS, 35 dB MPC threshold, delay/angular spreads and path gains, written to CSV |
| `subthz_channel_emulator.grc` | QT GUI emulator: choose LoS/NLoS, distance, realization and parameter set (published Table I or calibrated); shows the omni PDP, DoA/DoD angular power spectra and the spectrum of a 1 GHz test signal before and after the channel FIR filter |
| `build_flowgraphs.py` | writes both `.grc` files (keeps all block parameters consistent) |
| `export_canvas.py` | saves the GRC canvas of each flowgraph as PNG (`results/gnuradio/canvas_*.png`) |
| `screenshot_emulator.py` | runs the emulator off-screen and saves `results/gnuradio/emulator_*.png` |

## Paper step -> GNU Radio block

| Paper | Block |
|---|---|
| channel realization (Table I, Sec. VI generator) | embedded Python block `vna` |
| VNA sweep 145-146 GHz, 1001 tones, horns rotated 36 x 36 | same block, one complex vector of length 1001 per orientation, stream tags per channel |
| sounder noise | Noise Source -> Stream to Vector -> Add |
| Eq. 1, ADPS = \|IFFT(H)\|^2 / N_f | FFT (reverse, no window) -> Complex to Mag^2 -> Multiply Const |
| 35 dB threshold, spreads, path gain | embedded Python sink `adps_metrics` |
| channel emulation | Decimating FIR Filter with taps from a Function Probe |

## Running

```bash
# open in GNU Radio Companion
gnuradio-companion gnuradio/subthz_channel_emulator.grc

# batch: compile, run in parallel and analyse (results/gnuradio/)
python scripts/run_gnuradio.py --run --analyse --sets table1 calibrated
```

`scripts/run_gnuradio.py` uses `~/radioconda/python.exe` unless `GR_PYTHON` points elsewhere.
The generated `subthz_sounder_batch.py` can also be run directly:

```bash
python gnuradio/subthz_sounder_batch.py --condition nlos --n-channels 17 --distances measured --out-csv nlos.csv
```

`--params-file config/channel_parameters_calibrated.yaml` selects the calibrated parameter set
(made by `scripts/calibrate_model.py`; it is a fit to the paper's published synthetic results,
not the published Table I).
