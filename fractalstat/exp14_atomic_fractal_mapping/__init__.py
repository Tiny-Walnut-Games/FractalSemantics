"""
EXP-14: Atomic Fractal Mapping Module

This module provides comprehensive atomic fractal mapping capabilities for testing
whether electron shell structure naturally maps to fractal hierarchy.

CORRECTED DESIGN: Uses actual electron shell configuration as input, not naive Z-based mapping.

Tests whether atomic structure naturally emerges from fractal hierarchy.

Success Criteria:
- Fractal depth matches electron shell count (100% accuracy)
- Branching factor correlates with valence electrons (>0.95 correlation)
- Node count scales as branching^depth (exponential validation)
- Prediction errors decrease with shell depth (negative correlation)

Core Components:
- ElectronConfiguration: Electron shell configuration for elements
- ShellBasedFractalMapping: Mapping between electron shells and fractal parameters
- AtomicFractalMappingExperiment: Main experiment runner for atomic fractal mapping

Usage:
    from fractalstat.exp14_atomic_fractal_mapping import AtomicFractalMappingExperiment
    
    experiment = AtomicFractalMappingExperiment(
        elements_to_test=["hydrogen", "carbon", "gold"]
    )
    results = experiment.run()
"""

__version__ = "1.0.0"
__author__ = "FractalSemantics Team"
__description__ = "Atomic fractal mapping experiment for electron shell structure analysis"

# Import main classes for public API
from .entities import (
    ElectronConfiguration,
    ShellBasedFractalMapping,
)
from .experiment import AtomicFractalMappingExperiment

# Define what gets imported with "from exp14_atomic_fractal_mapping import *"
__all__ = [
    # Main experiment class
    "AtomicFractalMappingExperiment",
    
    # Core entities
    "ElectronConfiguration",
    "ShellBasedFractalMapping",
]