"""
EXP-17: Thermodynamic Validation of Fractal Systems - Data Entities

This module contains all the data structures and entities used in the thermodynamic
validation experiment. These entities represent the core components for testing
whether fractal simulations satisfy known thermodynamic equations.

If fractals are the fundamental structure of reality, they must obey ALL physical laws,
not just gravity. This experiment validates that fractal void/dense regions follow
thermodynamic principles.

Classes:
- ThermodynamicState: Thermodynamic properties of a fractal region
- ThermodynamicTransition: A transition between thermodynamic states
- ThermodynamicValidation: Results of thermodynamic law validation
"""

import json
import time
import secrets
import sys
import random
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
import statistics

secure_random = secrets.SystemRandom()

# ============================================================================
# EXP-17: THERMODYNAMIC MEASUREMENT STRUCTURES
# ============================================================================

@dataclass
class ThermodynamicState:
    """Thermodynamic properties of a fractal region."""

    region_id: str
    node_count: int
    total_energy: float
    average_cohesion: float
    entropy_estimate: float  # Information-theoretic entropy
    fractal_density: float
    temperature_proxy: float  # Based on interaction strength

    @property
    def energy_density(self) -> float:
        """Energy per node."""
        return self.total_energy / self.node_count if self.node_count > 0 else 0

    @property
    def information_density(self) -> float:
        """Information content per node."""
        # Based on fractal complexity and cohesion patterns
        return self.fractal_density * (1 + self.average_cohesion)


@dataclass
class ThermodynamicTransition:
    """A transition between thermodynamic states."""

    initial_state: ThermodynamicState
    final_state: ThermodynamicState
    work_done: float
    heat_transfer: float
    time_steps: int

    @property
    def delta_energy(self) -> float:
        """Change in total energy."""
        return self.final_state.total_energy - self.initial_state.total_energy

    @property
    def delta_entropy(self) -> float:
        """Change in entropy."""
        return self.final_state.entropy_estimate - self.initial_state.entropy_estimate


@dataclass
class ThermodynamicValidation:
    """Results of thermodynamic law validation."""

    law_tested: str
    description: str
    measured_value: float
    expected_range: Tuple[float, float]
    passed: bool
    confidence: float  # 0-1 scale

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{self.law_tested}: {status} ({self.measured_value:.4f} in {self.expected_range})"