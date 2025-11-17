"""
FractalStat - Complete validation suite for STAT7 7-dimensional addressing system
"""

__version__ = "0.1.0"
__author__ = "Tiny Walnut Games"

from fractalstat.stat7_entity import STAT7Coordinates, Realm, Horizon, Polarity
from fractalstat.stat7_experiments import BitChain, Coordinates

__all__ = [
    "STAT7Coordinates",
    "Realm",
    "Horizon",
    "Polarity",
    "BitChain",
    "Coordinates",
]
