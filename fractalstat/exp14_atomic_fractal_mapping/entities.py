"""
EXP-14: Atomic Fractal Mapping - Data Entities

This module contains all the data structures and entities used in the atomic fractal
mapping experiment. These entities represent the core components for mapping electron
shell configurations to fractal parameters and validating the structural relationships.

Classes:
- ElectronConfiguration: Electron shell configuration for elements
- ShellBasedFractalMapping: Mapping between electron shells and fractal parameters
"""

import json
import sys
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

# Import fractal hierarchy from EXP-13
from ..exp13_fractal_gravity import (
    get_element_fractal_density,
)

secure_random = np.random.RandomState(42)

# ============================================================================
# EXP-14 v2: ELECTRON SHELL DATA STRUCTURES
# ============================================================================

@dataclass
class ElectronConfiguration:
    """Electron shell configuration for elements."""

    element: str
    symbol: str
    atomic_number: int  # Z (protons)
    neutron_number: int  # N (neutrons)
    atomic_mass: float   # Atomic mass in u

    # Electron shell structure (THE KEY INPUT)
    electron_config: str  # Full configuration (e.g., "1s² 2s² 2p⁶ 3s² 3p⁶ 4s² 3d¹⁰")
    shell_count: int      # Number of electron shells (maps to fractal depth)
    valence_electrons: int  # Valence electrons (maps to branching factor)

    @property
    def noble_gas_core(self) -> str:
        """Get the noble gas core configuration."""
        # Simplified: return the appropriate noble gas core
        if self.atomic_number <= 2:
            return ""  # No core for H, He
        elif self.atomic_number <= 10:
            return "[He]"
        elif self.atomic_number <= 18:
            return "[Ne]"
        elif self.atomic_number <= 36:
            return "[Ar]"
        elif self.atomic_number <= 54:
            return "[Kr]"
        elif self.atomic_number <= 86:
            return "[Xe]"
        else:
            return "[Rn]"


@dataclass
class ShellBasedFractalMapping:
    """Mapping between electron shells and fractal parameters."""

    element: str
    electron_config: ElectronConfiguration

    # Direct mappings (no formulas - just count)
    fractal_depth: int        # = shell_count
    branching_factor: int     # = valence_electrons (+ nuclear adjustments)
    total_nodes: int          # = branching_factor ^ fractal_depth

    # Validation against EXP-13 observed densities
    predicted_density: float
    actual_density: float
    density_error: float

    # Structure validation (the real test)
    depth_matches_shells: bool     # fractal_depth == shell_count
    branching_matches_valence: bool # branching_factor ≈ valence_electrons
    node_growth_exponential: bool   # nodes = branching^depth