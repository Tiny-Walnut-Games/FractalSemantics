"""
EXP-17: Thermodynamic Validation of Fractal Systems

Tests whether fractal simulations satisfy known thermodynamic equations.

If fractals are the fundamental structure of reality, they must obey ALL physical laws,
not just gravity. This experiment validates that fractal void/dense regions follow
thermodynamic principles.

Success Criteria:
- Fractal void regions show minimum-entropy properties
- Fractal dense regions show maximum-entropy properties
- Energy conservation (1st Law) holds in fractal interactions
- Entropy increases over time (2nd Law) in fractal evolution
- Temperature equilibration (0th Law) occurs between fractal regions

Classes:
- ThermodynamicState: Thermodynamic properties of a fractal region
- ThermodynamicTransition: A transition between thermodynamic states
- ThermodynamicValidation: Results of thermodynamic law validation
- run_thermodynamic_validation_experiment: Main experiment runner

Usage:
    from fractalstat.exp17_thermodynamic_validation import run_thermodynamic_validation_experiment

    results = run_thermodynamic_validation_experiment()
    print(f"Thermodynamic consistency: {results['summary']['overall_success']}")
"""

__version__ = "1.0.0"
__author__ = "Tiny Walnut Games"
__description__ = "Thermodynamic Validation of Fractal Systems"

from .entities import (
    ThermodynamicState,
    ThermodynamicTransition,
    ThermodynamicValidation,
)
from .experiment import (
    measure_fractal_entropy,
    measure_fractal_energy,
    measure_fractal_temperature,
    create_fractal_region,
    validate_first_law,
    validate_second_law,
    validate_zeroth_law,
    validate_fractal_void_density,
    run_thermodynamic_validation_experiment,
    save_results,
)

__all__ = [
    # Core entities
    'ThermodynamicState',
    'ThermodynamicTransition',
    'ThermodynamicValidation',

    # Measurement functions
    'measure_fractal_entropy',
    'measure_fractal_energy',
    'measure_fractal_temperature',
    'create_fractal_region',

    # Validation functions
    'validate_first_law',
    'validate_second_law',
    'validate_zeroth_law',
    'validate_fractal_void_density',

    # Main experiment
    'run_thermodynamic_validation_experiment',
    'save_results',
]