"""
EXP-18: Falloff Injection in Thermodynamics

Tests whether applying the same falloff formula used in gravity to thermodynamic
measurements makes fractal thermodynamics behave more like classical thermodynamics.

If gravity and thermodynamics both emerge from fractal structure, then injecting
the same falloff should make thermodynamic behavior more "classical" (energy conserved,
entropy increasing, temperatures equilibrating).

Success Criteria:
- With falloff injection, energy conservation improves
- With falloff injection, entropy shows classical increase
- With falloff injection, temperature equilibration occurs
- With falloff injection, void/dense entropy follows classical expectations

Classes:
- run_falloff_thermodynamics_experiment: Main experiment runner

Usage:
    from fractalstat.exp18_falloff_thermodynamics import run_falloff_thermodynamics_experiment

    results = run_falloff_thermodynamics_experiment(falloff_exponent=2.0)
    print(f"Falloff improves thermodynamics: {results['comparison']['falloff_improves_thermodynamics']}")
"""

__version__ = "1.0.0"
__author__ = "Tiny Walnut Games"
__description__ = "Falloff Injection in Thermodynamics"

from .experiment import (
    measure_fractal_energy_with_falloff,
    measure_fractal_entropy_with_falloff,
    measure_fractal_temperature_with_falloff,
    create_fractal_region_with_falloff,
    run_falloff_thermodynamics_experiment,
    save_results,
)

__all__ = [
    # Measurement functions with falloff
    'measure_fractal_energy_with_falloff',
    'measure_fractal_entropy_with_falloff',
    'measure_fractal_temperature_with_falloff',
    'create_fractal_region_with_falloff',

    # Main experiment
    'run_falloff_thermodynamics_experiment',
    'save_results',
]