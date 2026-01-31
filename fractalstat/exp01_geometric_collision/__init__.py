"""
EXP-01: Geometric Collision Resistance Test - Modular Implementation

This module provides a modular implementation of the geometric collision resistance test
that validates FractalStat coordinates achieve collision resistance through semantic
differentiation rather than coordinate space geometry.

Core Hypothesis:
FractalStat 8D coordinate space demonstrates perfect collision resistance where:
- 2D/3D coordinate subspaces show expected collisions when exceeding space bounds
- 4D+ coordinate subspaces exhibit geometric collision resistance
- The 8th dimension (alignment) provides complete expressivity coverage
- Collision resistance is purely mathematical, cryptography serves as assurance

Usage:
    from fractalstat.exp01_geometric_collision import EXP01_GeometricCollisionResistance
    
    experiment = EXP01_GeometricCollisionResistance(sample_size=100000)
    results, success = experiment.run()
"""

from .entities import EXP01_Result
from .experiment import EXP01_GeometricCollisionResistance

__all__ = [
    'EXP01_Result',
    'EXP01_GeometricCollisionResistance'
]