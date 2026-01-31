"""
EXP-15: Topological Conservation Laws

Tests whether fractal systems conserve topology (hierarchical structure, connectivity,
branching patterns) rather than classical energy and momentum.

CORE HYPOTHESIS:
In fractal physics, topology is the conserved quantity, not energy.
Classical Newtonian mechanics conserves energy but not topology.
Fractal mechanics conserves topology but not energy.

PHASES:
1. Define topological invariants (node count, depth, connectivity, branching)
2. Run orbital dynamics simulation and check conservation over time
3. Compare against classical Newtonian conservation laws
4. Prove topology conserved while energy is not

SUCCESS CRITERIA:
- Topology conserved over 1-year orbit (100% stability)
- Classical energy shows drift (non-conservation)
- Node count, depth, connectivity remain invariant
- Address collisions remain zero
- Structure entropy stays constant

This module provides:
- Topological invariants computation and analysis
- Classical conservation law analysis
- Orbital dynamics with topological tracking
- Experiment orchestration and validation
- Results processing and file I/O

Usage:
    from fractalstat.exp15_topological_conservation import TopologicalConservationExperiment

    experiment = TopologicalConservationExperiment()
    results = experiment.run()
"""

from .entities import (
    TopologicalInvariants,
    TopologicalConservationMeasurement,
    TopologicalConservationAnalysis,
    ClassicalConservationAnalysis,
)
from .experiment import (
    TopologicalConservationExperiment,
    TopologicalConservationTestResult,
    compute_topological_invariants,
    compare_topological_invariants,
    compute_classical_conservation,
    integrate_orbit_with_topological_tracking,
    save_results,
)

__all__ = [
    # Core entities
    "TopologicalInvariants",
    "TopologicalConservationMeasurement",
    "TopologicalConservationAnalysis",
    "ClassicalConservationAnalysis",
    
    # Experiment classes
    "TopologicalConservationExperiment",
    "TopologicalConservationTestResult",
    
    # Core functions
    "compute_topological_invariants",
    "compare_topological_invariants",
    "compute_classical_conservation",
    "integrate_orbit_with_topological_tracking",
    "save_results",
]

__version__ = "1.0.0"