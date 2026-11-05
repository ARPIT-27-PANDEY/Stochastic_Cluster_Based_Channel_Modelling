#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Sub-THz 145 GHz D2D channel sounder (batch validation)
# Author: Group 10 - P. Bhaskar, A. Tripathi, A. K. Pandey
# Description: GNU Radio implementation of the measurement chain of Cheng et al. (IEEE TWC 2026): cluster-model channel -> per-orientation 1001-tone VNA sweep -> sounder noise -> IFFT -> |.|^2/N_f (Eq. 1) -> ADPS, 35 dB MPC threshold, delay/angular spreads and path gains (Sec. VIII). Run from the command line: python subthz_sounder_batch.py --condition nlos --n-channels 17 --out-csv out.csv
# GNU Radio version: 3.10.12.0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import numpy as np
import subthz_sounder_batch_adps_metrics as adps_metrics  # embedded python block
import subthz_sounder_batch_vna as vna  # embedded python block
import threading




class subthz_sounder_batch(gr.top_block):

    def __init__(self, condition='los', n_channels=4, start_index=0, seed=20260723, distances='measured', params_file='', snr_db=60.0, out_csv='sounder.csv'):
        gr.top_block.__init__(self, "Sub-THz 145 GHz D2D channel sounder (batch validation)", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.condition = condition
        self.n_channels = n_channels
        self.start_index = start_index
        self.seed = seed
        self.distances = distances
        self.params_file = params_file
        self.snr_db = snr_db
        self.out_csv = out_csv

        ##################################################
        # Variables
        ##################################################
        self.nf = nf = 1001
        self.dr_db = dr_db = 35.0

        ##################################################
        # Blocks
        ##################################################

        self.vna = vna.blk(condition=condition, n_channels=n_channels, start_index=start_index, seed=seed, distances=distances, params_file=params_file)
        self.rx_add = blocks.add_vcc(nf)
        self.norm = blocks.multiply_const_vff([1.0 / nf] * nf)
        self.noise_vec = blocks.stream_to_vector(gr.sizeof_gr_complex*1, nf)
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, (10 ** (-snr_db / 20)), 1)
        self.mag2 = blocks.complex_to_mag_squared(nf)
        self.ifft = fft.fft_vcc(nf, False, [], False, 1)
        self.adps_metrics = adps_metrics.blk(out_csv=out_csv, condition=condition, dr_db=dr_db)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.ifft, 0), (self.mag2, 0))
        self.connect((self.mag2, 0), (self.norm, 0))
        self.connect((self.noise, 0), (self.noise_vec, 0))
        self.connect((self.noise_vec, 0), (self.rx_add, 1))
        self.connect((self.norm, 0), (self.adps_metrics, 0))
        self.connect((self.rx_add, 0), (self.ifft, 0))
        self.connect((self.vna, 0), (self.rx_add, 0))


    def get_condition(self):
        return self.condition

    def set_condition(self, condition):
        self.condition = condition
        self.vna.condition = self.condition
        self.adps_metrics.condition = self.condition

    def get_n_channels(self):
        return self.n_channels

    def set_n_channels(self, n_channels):
        self.n_channels = n_channels
        self.vna.n_channels = self.n_channels

    def get_start_index(self):
        return self.start_index

    def set_start_index(self, start_index):
        self.start_index = start_index
        self.vna.start_index = self.start_index

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        self.vna.seed = self.seed

    def get_distances(self):
        return self.distances

    def set_distances(self, distances):
        self.distances = distances
        self.vna.distances = self.distances

    def get_params_file(self):
        return self.params_file

    def set_params_file(self, params_file):
        self.params_file = params_file
        self.vna.params_file = self.params_file

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.noise.set_amplitude((10 ** (-self.snr_db / 20)))

    def get_out_csv(self):
        return self.out_csv

    def set_out_csv(self, out_csv):
        self.out_csv = out_csv
        self.adps_metrics.out_csv = self.out_csv

    def get_nf(self):
        return self.nf

    def set_nf(self, nf):
        self.nf = nf
        self.norm.set_k([1.0 / self.nf] * self.nf)

    def get_dr_db(self):
        return self.dr_db

    def set_dr_db(self, dr_db):
        self.dr_db = dr_db
        self.adps_metrics.dr_db = self.dr_db



def argument_parser():
    description = 'GNU Radio implementation of the measurement chain of Cheng et al. (IEEE TWC 2026): cluster-model channel -> per-orientation 1001-tone VNA sweep -> sounder noise -> IFFT -> |.|^2/N_f (Eq. 1) -> ADPS, 35 dB MPC threshold, delay/angular spreads and path gains (Sec. VIII). Run from the command line: python subthz_sounder_batch.py --condition nlos --n-channels 17 --out-csv out.csv'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--condition", dest="condition", type=str, default='los',
        help="Set Condition (los / nlos) [default=%(default)r]")
    parser.add_argument(
        "--n-channels", dest="n_channels", type=intx, default=4,
        help="Set Number of channel realizations [default=%(default)r]")
    parser.add_argument(
        "--start-index", dest="start_index", type=intx, default=0,
        help="Set First realization index [default=%(default)r]")
    parser.add_argument(
        "--seed", dest="seed", type=intx, default=20260723,
        help="Set Seed [default=%(default)r]")
    parser.add_argument(
        "--distances", dest="distances", type=str, default='measured',
        help="Set Link distances (measured / uniform) [default=%(default)r]")
    parser.add_argument(
        "--params-file", dest="params_file", type=str, default='',
        help="Set Parameter YAML ('' = Table I) [default=%(default)r]")
    parser.add_argument(
        "--snr-db", dest="snr_db", type=eng_float, default=eng_notation.num_to_str(float(60.0)),
        help="Set Sounder SNR per tone (dB below strongest tap) [default=%(default)r]")
    parser.add_argument(
        "--out-csv", dest="out_csv", type=str, default='sounder.csv',
        help="Set Output CSV [default=%(default)r]")
    return parser


def main(top_block_cls=subthz_sounder_batch, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(condition=options.condition, n_channels=options.n_channels, start_index=options.start_index, seed=options.seed, distances=options.distances, params_file=options.params_file, snr_db=options.snr_db, out_csv=options.out_csv)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    tb.wait()


if __name__ == '__main__':
    main()
