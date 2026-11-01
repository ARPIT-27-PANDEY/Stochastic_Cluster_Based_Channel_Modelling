"""Python reproduction of the 145 GHz sub-THz outdoor D2D cluster channel model.

Cheng, Abbasi, Gomez-Ponce, Zhang, Molisch, IEEE Trans. Wireless Commun., vol. 25, 2026.
"""
from .adps import DelayAngleGrid
from .channel_generator import ChannelGenerator, ChannelRealization
from .inverse_mapping import DiscretizationCorrection, invert_spreads, solve_decay_constants
from .parameters import load_channel_parameters, load_simulation_config

__all__ = [
    "ChannelGenerator",
    "ChannelRealization",
    "DelayAngleGrid",
    "DiscretizationCorrection",
    "invert_spreads",
    "solve_decay_constants",
    "load_channel_parameters",
    "load_simulation_config",
]
__version__ = "1.0.0"
