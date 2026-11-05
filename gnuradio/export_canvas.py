"""Export the GNU Radio Companion canvas of each flowgraph to PNG (GRC's own screen-capture code).

    <radioconda>/python gnuradio/export_canvas.py

Writes results/gnuradio/canvas_<flowgraph>.png.
"""
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gtk  # noqa: E402,F401
from gnuradio import gr  # noqa: E402
from gnuradio.grc.gui.Platform import Platform  # noqa: E402  (import first: avoids a circular import)
from gnuradio.grc.gui import Utils  # noqa: E402
from gnuradio.grc.gui.canvas import flowgraph as _canvas_fg  # noqa: E402


class _NoMenu:  # no main window here, so no right-click menu is needed
    def __init__(self, *a, **k):
        pass

    def popup(self, *a, **k):
        pass


_canvas_fg._ContextMenu = _NoMenu

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "gnuradio"
OUT.mkdir(parents=True, exist_ok=True)

_app = Gtk.Application(application_id='org.subthz.canvasexport')
_app.set_default()   # GRC's canvas FlowGraph looks up the default Gtk.Application

platform = Platform(version=gr.version(), version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()),
                    prefs=gr.prefs(), install_prefix=gr.prefix())
platform.build_library()
for name in sys.argv[1:] or ["subthz_sounder_batch", "subthz_channel_emulator"]:
    fg = platform.make_flow_graph(str(HERE / f"{name}.grc"))
    fg.rewrite()
    fg.validate()
    import cairo
    _cr = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10))
    fg.update_elements_to_draw()
    fg.create_labels(_cr)      # labels/shapes define the extents used for the image size
    fg.create_shapes()
    out = OUT / f"canvas_{name}.png"
    Utils.make_screenshot(fg, str(out))
    print("saved", out)
