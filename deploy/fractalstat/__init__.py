"""FractalStat.

A Complete validation suite for multi-dimensional addressing-systems.
===============
A comprehensive suite of tests to ensure reliability, performance,
and robustness of the FractalStat multi-dimensional addressing system.

    8-dimensional addressing space for all entities with 100% expressivity.

    Each dimension represents a different axis of entity existence:
      1. Realm: Domain/type classification
      2. Lineage: Generation or tier progression from LUCA
      3. Adjacency: Semantic/functional proximity score (0-100)
      4. Horizon: Lifecycle stage
      5. Luminosity: Activity level (0-100)
      6. Polarity: Resonance/affinity type
      7. Dimensionality: Fractal depth / detail level
      8. Alignment: Social/coordination dynamics (NEW - 100% expressivity boost)
"""

from fractalstat.dynamic_enum import Realm, Horizon, Polarity, Alignment
from fractalstat.fractalstat_entity import FractalStatCoordinates, FractalStatEntity
from fractalstat.fractalstat_entity import BitChain
from fractalstat.fractalstat_rag_bridge import FractalStatRAGBridge

__version__ = "0.1.0"
__author__ = "Tiny Walnut Games"

# ============================================================================
# 8D FractalStat Dimension Constants for API Consistency
# ============================================================================

# Dimension Names (for consistent API usage across the project)
REALM = "realm"
LINEAGE = "lineage"
ADJACENCY = "adjacency"
HORIZON = "horizon"
LUMINOSITY = "luminosity"
POLARITY = "polarity"
DIMENSIONALITY = "dimensionality"
ALIGNMENT = "alignment"

# Dimension Order (for consistent iteration/serialization)
FRACTALSTAT_DIMENSIONS = [
    REALM,
    LINEAGE,
    ADJACENCY,
    HORIZON,
    LUMINOSITY,
    POLARITY,
    DIMENSIONALITY,
    ALIGNMENT,
]

__all__ = [
    # Enums
    "Realm",
    "Horizon",
    "Polarity",
    "Alignment",

    # Dimension Constants
    "REALM",
    "LINEAGE",
    "ADJACENCY",
    "HORIZON",
    "LUMINOSITY",
    "POLARITY",
    "DIMENSIONALITY",
    "ALIGNMENT",

    # Dimension List
    "FRACTALSTAT_DIMENSIONS",

    # Classes
    "FractalStatCoordinates",
    "FractalStatEntity",
    "BitChain",

    # RAG Bridge
    "FractalStatRAGBridge",
]
