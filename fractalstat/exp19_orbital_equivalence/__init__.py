"""
EXP-19: Orbital Equivalence - Prove orbital mechanics ≈ fractal cohesion mechanics

Tests whether orbital mechanics and fractal cohesion mechanics are equivalent representations
of the same physical reality. If Newton's gravity emerges from fractal topology, then
orbital mechanics and fractal calculations should produce identical trajectories.

This module provides:
- CelestialBody, OrbitalSystem: Classical Newtonian mechanics entities
- FractalBody, FractalOrbitalSystem: Fractal mechanics entities  
- simulate_orbital_trajectory: Classical orbital simulation
- simulate_fractal_trajectory: Fractal cohesion simulation
- TrajectoryComparison: Comparison between frameworks
- OrbitalEquivalenceTest: Main equivalence testing
- run_orbital_equivalence_test: Main experiment execution

Usage:
    from fractalstat.exp19_orbital_equivalence import run_orbital_equivalence_test, OrbitalEquivalenceTest

    # Test Earth-Sun system equivalence
    result = run_orbital_equivalence_test(system_name="Earth-Sun", simulation_time=365*24*3600)
    print(f"Equivalence confirmed: {result.equivalence_confirmed}")

    # Test Solar System with perturbations
    result = run_orbital_equivalence_test(
        system_name="Solar System", 
        simulation_time=365*24*3600,
        include_perturbation=True
    )
"""

from .entities import CelestialBody, OrbitalSystem, FractalBody, FractalOrbitalSystem
from .classical_mechanics import gravitational_force, orbital_acceleration, simulate_orbital_trajectory
from .fractal_mechanics import fractal_cohesion_force, simulate_fractal_trajectory
from .comparison import TrajectoryComparison, OrbitalEquivalenceTest
from .systems import create_earth_sun_system, create_solar_system, add_rogue_planet_perturbation
from .experiment import run_orbital_equivalence_test
from .results import save_results

__all__ = [
    "CelestialBody",
    "OrbitalSystem", 
    "FractalBody",
    "FractalOrbitalSystem",
    "gravitational_force",
    "orbital_acceleration",
    "simulate_orbital_trajectory",
    "fractal_cohesion_force", 
    "simulate_fractal_trajectory",
    "TrajectoryComparison",
    "OrbitalEquivalenceTest",
    "create_earth_sun_system",
    "create_solar_system",
    "add_rogue_planet_perturbation",
    "run_orbital_equivalence_test",
    "save_results",
]