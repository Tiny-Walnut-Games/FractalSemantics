"""
EXP-20: Vector Field Derivation from Fractal Hierarchy

This module provides the complete implementation for deriving vector fields
from fractal hierarchy structures, enabling the reproduction of Newtonian
gravity and orbital mechanics through fractal physics.

Core Components:
- FractalEntity: Entities with fractal properties for vector field derivation
- VectorFieldDerivationSystem: System for testing different derivation approaches
- OrbitalTrajectory: Trajectory computed using vector field integration
- EXP20_VectorFieldResults: Complete experiment results

Usage:
    from fractalstat.exp20_vector_field_derivation import run_exp20_vector_field_derivation

    results = run_exp20_vector_field_derivation()
    print(f"Best approach: {results.best_approach}")
    print(f"Model complete: {results.model_complete}")
"""

from .entities import FractalEntity, create_earth_sun_fractal_entities
from .vector_field_system import (
    VectorFieldDerivationSystem, 
    VectorFieldApproach,
    compute_force_vector_via_branching,
    compute_force_vector_via_branching_difference,
    compute_force_vector_via_branching_normalized
)
from .trajectory import OrbitalTrajectory, TrajectoryComparison, integrate_orbit_with_vector_field
from .experiment import run_exp20_vector_field_derivation, EXP20_VectorFieldResults
from .validation import validate_inverse_square_law_for_approach

__all__ = [
    'FractalEntity',
    'create_earth_sun_fractal_entities',
    'VectorFieldDerivationSystem',
    'VectorFieldApproach',
    'compute_force_vector_via_branching',
    'compute_force_vector_via_branching_difference',
    'compute_force_vector_via_branching_normalized',
    'OrbitalTrajectory',
    'TrajectoryComparison',
    'integrate_orbit_with_vector_field',
    'run_exp20_vector_field_derivation',
    'EXP20_VectorFieldResults',
    'validate_inverse_square_law_for_approach'
]
