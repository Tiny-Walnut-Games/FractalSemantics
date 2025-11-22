"""FractalStat.

A Complete validation suite for multi-dimensional addressing-systems.
===============
A comprehensive suite of tests to ensure reliability, performance,
and robustness of the FractalStat multi-dimensional addressing system.
"""

__version__ = "0.1.0"
__author__ = "Tiny Walnut Games"

from fractalstat.fractalstat_entity import Realm, Horizon, Polarity, Alignment, FractalStatCoordinates, FractalStatEntity
from fractalstat.fractalstat_experiments import BitChain

__all__ = [
    "Realm",
    "Horizon",
    "Polarity",
    "Alignment",
    "FractalStatCoordinates",
    "FractalStatEntity",
    "BitChain",
]
