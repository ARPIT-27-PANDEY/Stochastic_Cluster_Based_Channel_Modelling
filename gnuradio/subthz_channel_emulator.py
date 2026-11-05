#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Sub-THz 145 GHz D2D cluster channel emulator
# Author: Group 10 - P. Bhaskar, A. Tripathi, A. K. Pandey
# Description: Interactive GNU Radio emulator of the 145 GHz outdoor D2D cluster model (Cheng et al., IEEE TWC 2026): draws double-directional channels from Table I, shows the omni PDP and the DoA/DoD angular power spectra, and passes a 1 GHz-wide test signal through the channel impulse response with the GNU Radio FIR filter.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import analog
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
import numpy as np
import sip
import subthz_channel_emulator_emu as emu  # embedded python block
import threading
import time



class subthz_channel_emulator(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Sub-THz 145 GHz D2D cluster channel emulator", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Sub-THz 145 GHz D2D cluster channel emulator")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "subthz_channel_emulator")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.stats_probe = stats_probe = 'drawing channel...'
        self.taps_probe = taps_probe = [1.0]
        self.stats = stats = stats_probe
        self.seed = seed = 20260723
        self.samp_rate = samp_rate = 1e9
        self.model = model = ''
        self.index = index = 0
        self.d_m = d_m = 20
        self.condition = condition = 'los'

        ##################################################
        # Blocks
        ##################################################

        self._index_range = qtgui.Range(0, 999, 1, 0, 200)
        self._index_win = qtgui.RangeWidget(self._index_range, self.set_index, "Realization #", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._index_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        # Create the options list
        self._model_options = ['', 'config/channel_parameters_calibrated.yaml']
        # Create the labels list
        self._model_labels = ['Table I (published)', 'Calibrated']
        # Create the combo box
        # Create the radio buttons
        self._model_group_box = Qt.QGroupBox("Parameter set" + ": ")
        self._model_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._model_button_group = variable_chooser_button_group()
        self._model_group_box.setLayout(self._model_box)
        for i, _label in enumerate(self._model_labels):
            radio_button = Qt.QRadioButton(_label)
            self._model_box.addWidget(radio_button)
            self._model_button_group.addButton(radio_button, i)
        self._model_callback = lambda i: Qt.QMetaObject.invokeMethod(self._model_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._model_options.index(i)))
        self._model_callback(self.model)
        self._model_button_group.buttonClicked[int].connect(
            lambda i: self.set_model(self._model_options[i]))
        self.top_grid_layout.addWidget(self._model_group_box, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._d_m_range = qtgui.Range(1, 100, 0.5, 20, 200)
        self._d_m_win = qtgui.RangeWidget(self._d_m_range, self.set_d_m, "Tx-Rx distance d_LoS (m)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._d_m_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        # Create the options list
        self._condition_options = ['los', 'nlos']
        # Create the labels list
        self._condition_labels = ['LoS', 'NLoS']
        # Create the combo box
        # Create the radio buttons
        self._condition_group_box = Qt.QGroupBox("Propagation condition" + ": ")
        self._condition_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._condition_button_group = variable_chooser_button_group()
        self._condition_group_box.setLayout(self._condition_box)
        for i, _label in enumerate(self._condition_labels):
            radio_button = Qt.QRadioButton(_label)
            self._condition_box.addWidget(radio_button)
            self._condition_button_group.addButton(radio_button, i)
        self._condition_callback = lambda i: Qt.QMetaObject.invokeMethod(self._condition_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._condition_options.index(i)))
        self._condition_callback(self.condition)
        self._condition_button_group.buttonClicked[int].connect(
            lambda i: self.set_condition(self._condition_options[i]))
        self.top_grid_layout.addWidget(self._condition_group_box, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.emu = emu.blk(condition=condition, d_m=d_m, index=index, seed=seed, params_file=model)
        def _taps_probe_probe():
          self.flowgraph_started.wait()
          while True:

            val = self.emu.taps()
            try:
              try:
                self.doc.add_next_tick_callback(functools.partial(self.set_taps_probe,val))
              except AttributeError:
                self.set_taps_probe(val)
            except AttributeError:
              pass
            time.sleep(1.0 / (2))
        _taps_probe_thread = threading.Thread(target=_taps_probe_probe)
        _taps_probe_thread.daemon = True
        _taps_probe_thread.start()
        self.tx_sig = analog.noise_source_c(analog.GR_GAUSSIAN, 1, 3)
        self.thr_vec = blocks.throttle( gr.sizeof_float*1001, 10, True, 0 if "auto" == "auto" else max( int(float(0.1) * 10) if "auto" == "time" else int(0.1), 1) )
        self.thr_sig = blocks.throttle( gr.sizeof_gr_complex*1, 2000000.0, True, 0 if "auto" == "auto" else max( int(float(0.1) * 2000000.0) if "auto" == "time" else int(0.1), 1) )
        def _stats_probe_probe():
          self.flowgraph_started.wait()
          while True:

            val = self.emu.summary()
            try:
              try:
                self.doc.add_next_tick_callback(functools.partial(self.set_stats_probe,val))
              except AttributeError:
                self.set_stats_probe(val)
            except AttributeError:
              pass
            time.sleep(1.0 / (2))
        _stats_probe_thread = threading.Thread(target=_stats_probe_probe)
        _stats_probe_thread.daemon = True
        _stats_probe_thread.start()
        self._stats_tool_bar = Qt.QToolBar(self)

        if None:
            self._stats_formatter = None
        else:
            self._stats_formatter = lambda x: str(x)

        self._stats_tool_bar.addWidget(Qt.QLabel("Channel"))
        self._stats_label = Qt.QLabel(str(self._stats_formatter(self.stats)))
        self._stats_tool_bar.addWidget(self._stats_label)
        self.top_grid_layout.addWidget(self._stats_tool_bar, 2, 0, 1, 2)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.spec = qtgui.freq_sink_c(
            4096, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            145.5e9, #fc
            samp_rate, #bw
            "1 GHz test signal: transmitted vs after the channel (145-146 GHz)", #name
            2,
            None # parent
        )
        self.spec.set_update_time(0.20)
        self.spec.set_y_axis((-60), (-20))
        self.spec.set_y_label('Relative Gain', 'dB')
        self.spec.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.spec.enable_autoscale(False)
        self.spec.enable_grid(True)
        self.spec.set_fft_average(0.05)
        self.spec.enable_axis_labels(True)
        self.spec.enable_control_panel(False)
        self.spec.set_fft_window_normalized(False)



        labels = ['Tx (white)', 'Rx (through channel)', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.spec.set_line_label(i, "Data {0}".format(i))
            else:
                self.spec.set_line_label(i, labels[i])
            self.spec.set_line_width(i, widths[i])
            self.spec.set_line_color(i, colors[i])
            self.spec.set_line_alpha(i, alphas[i])

        self._spec_win = sip.wrapinstance(self.spec.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._spec_win, 4, 1, 1, 1)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.pdp_sink = qtgui.vector_sink_f(
            1001,
            0,
            1,
            "Excess delay (ns)",
            "Power (dB)",
            "Omni PDP (max over orientation pairs), 35 dB MPC range",
            1, # Number of inputs
            None # parent
        )
        self.pdp_sink.set_update_time(0.2)
        self.pdp_sink.set_y_axis((-200), (-40))
        self.pdp_sink.enable_autoscale(True)
        self.pdp_sink.enable_grid(True)
        self.pdp_sink.set_x_axis_units("")
        self.pdp_sink.set_y_axis_units("")
        self.pdp_sink.set_ref_level(0)


        labels = ["omni PDP", '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.pdp_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.pdp_sink.set_line_label(i, labels[i])
            self.pdp_sink.set_line_width(i, widths[i])
            self.pdp_sink.set_line_color(i, colors[i])
            self.pdp_sink.set_line_alpha(i, alphas[i])

        self._pdp_sink_win = sip.wrapinstance(self.pdp_sink.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._pdp_sink_win, 3, 0, 1, 2)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.chan_fir = filter.fir_filter_ccc(1, taps_probe)
        self.chan_fir.declare_sample_delay(0)
        self.aps_sink = qtgui.vector_sink_f(
            36,
            (-180),
            10,
            "Angle (deg)",
            "Power (dB)",
            "Angular power spectra",
            2, # Number of inputs
            None # parent
        )
        self.aps_sink.set_update_time(0.2)
        self.aps_sink.set_y_axis((-200), (-40))
        self.aps_sink.enable_autoscale(True)
        self.aps_sink.enable_grid(True)
        self.aps_sink.set_x_axis_units("")
        self.aps_sink.set_y_axis_units("")
        self.aps_sink.set_ref_level(0)


        labels = ["DoA (Rx)", "DoD (Tx)", '', '', '',
            '', '', '', '', '']
        widths = [2, 2, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["red", "dark green", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.aps_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.aps_sink.set_line_label(i, labels[i])
            self.aps_sink.set_line_width(i, widths[i])
            self.aps_sink.set_line_color(i, colors[i])
            self.aps_sink.set_line_alpha(i, alphas[i])

        self._aps_sink_win = sip.wrapinstance(self.aps_sink.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._aps_sink_win, 4, 0, 1, 1)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.chan_fir, 0), (self.spec, 1))
        self.connect((self.emu, 1), (self.aps_sink, 0))
        self.connect((self.emu, 2), (self.aps_sink, 1))
        self.connect((self.emu, 0), (self.thr_vec, 0))
        self.connect((self.thr_sig, 0), (self.chan_fir, 0))
        self.connect((self.thr_sig, 0), (self.spec, 0))
        self.connect((self.thr_vec, 0), (self.pdp_sink, 0))
        self.connect((self.tx_sig, 0), (self.thr_sig, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "subthz_channel_emulator")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_stats_probe(self):
        return self.stats_probe

    def set_stats_probe(self, stats_probe):
        self.stats_probe = stats_probe
        self.set_stats(self.stats_probe)

    def get_taps_probe(self):
        return self.taps_probe

    def set_taps_probe(self, taps_probe):
        self.taps_probe = taps_probe
        self.chan_fir.set_taps(self.taps_probe)

    def get_stats(self):
        return self.stats

    def set_stats(self, stats):
        self.stats = stats
        Qt.QMetaObject.invokeMethod(self._stats_label, "setText", Qt.Q_ARG("QString", str(self._stats_formatter(self.stats))))

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        self.emu.seed = self.seed

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.spec.set_frequency_range(145.5e9, self.samp_rate)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        self._model_callback(self.model)
        self.emu.params_file = self.model

    def get_index(self):
        return self.index

    def set_index(self, index):
        self.index = index
        self.emu.index = self.index

    def get_d_m(self):
        return self.d_m

    def set_d_m(self, d_m):
        self.d_m = d_m
        self.emu.d_m = self.d_m

    def get_condition(self):
        return self.condition

    def set_condition(self, condition):
        self.condition = condition
        self._condition_callback(self.condition)
        self.emu.condition = self.condition




def main(top_block_cls=subthz_channel_emulator, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
