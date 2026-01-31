"""
EXP-04: FractalStat Fractal Scaling Test

Tests whether FractalStat addressing maintains consistency and zero collisions
when scaled from 1K → 10K → 100K → 1M data points.

Verifies the "fractal" property: self-similar behavior at all scales.

Status: Phase 2 validation experiment
"""

from .entities import (
    ScaleTestConfig,
    ScaleTestResults, 
    FractalScalingResults,
    generate_random_bitchain,
    secure_random
)
from .experiment import (
    run_scale_test,
    analyze_degradation,
    run_fractal_scaling_test
)
from .results import save_results

__version__ = "1.0.0"
__all__ = [
    "ScaleTestConfig",
    "ScaleTestResults", 
    "FractalScalingResults",
    "run_scale_test",
    "analyze_degradation", 
    "run_fractal_scaling_test",
    "save_results"
]