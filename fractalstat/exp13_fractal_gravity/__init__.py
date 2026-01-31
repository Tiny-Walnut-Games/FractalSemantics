"""
EXP-13: Fractal Gravity Module

This module provides comprehensive fractal gravity testing capabilities for evaluating
whether fractal entities naturally create gravitational cohesion without falloff,
and whether injecting falloff produces consistent weakening across all element types.

Core Postulates:
1. Fractal Cohesion Without Falloff - cohesion constant across hierarchical levels
2. Elements as Fractal Constructs - mass = fractal density (hierarchical complexity)
3. Universal Interaction Mechanism - same falloff pattern for all elements
4. Hierarchical Distance is Fundamental - topology, not space, determines interactions

Hypothesis:
Natural cohesion depends ONLY on hierarchical relationship (constant, no falloff).
Falloff injection produces identical mathematical patterns across all elements.

Main Components:
- FractalGravityExperiment: Main experiment runner for fractal gravity testing
- FractalNode: A node in a pure fractal hierarchy (no spatial coordinates)
- FractalHierarchy: A pure fractal tree structure for a single element type
- HierarchicalCohesionMeasurement: Records cohesion between nodes at specific distances
- ElementGravityResults: Results for a single element's gravitational behavior
- EXP13v2_GravityTestResults: Complete results from fractal gravity test

Usage:
    from fractalstat.exp13_fractal_gravity import FractalGravityExperiment
    
    experiment = FractalGravityExperiment(
        elements_to_test=["gold", "nickel", "copper"],
        max_hierarchy_depth=5,
        interaction_samples=5000
    )
    results = experiment.run()
"""

__version__ = "1.0.0"
__author__ = "FractalSemantics Team"
__description__ = "Fractal gravity experiment for hierarchical cohesion testing"

# Import main classes for public API
from .entities import (
    FractalNode,
    FractalHierarchy,
    HierarchicalCohesionMeasurement,
    ElementGravityResults,
    EXP13v2_GravityTestResults,
)
from .experiment import (
    FractalGravityExperiment,
    compute_natural_cohesion,
)

# Define what gets imported with "from exp13_fractal_gravity import *"
__all__ = [
    # Main experiment class
    "FractalGravityExperiment",
    
    # Core entities
    "FractalNode",
    "FractalHierarchy",
    "HierarchicalCohesionMeasurement",
    "ElementGravityResults",
    "EXP13v2_GravityTestResults",
    
    # Utility functions
    "compute_natural_cohesion",
]
