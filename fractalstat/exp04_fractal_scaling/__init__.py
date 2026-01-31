"""
EXP-04: Bit-Chain FractalStat Fractal Scaling Test - Modular Implementation

This module provides a modular implementation of the fractal scaling test
that verifies whether FractalStat addressing maintains consistency and zero collisions
when scaled from 1K → 10K → 100K → 1M data points.

Core Hypothesis:
FractalStat addressing maintains the "fractal" property: self-similar behavior
at all scales with zero collisions and logarithmic retrieval performance.

Methodology:
1. Generate bit-chains at increasing scale levels (1K, 10K, 100K, 1M)
2. Compute addresses and verify zero collisions at each scale
3. Test retrieval performance and measure latency scaling
4. Analyze degradation patterns to confirm fractal properties
5. Validate that system behavior is self-similar across scales

Success Criteria:
- Zero collisions at all scale levels
- Retrieval latency scales logarithmically (not linearly)
- Address generation throughput remains consistent
- System behavior is self-similar across scales

Usage:
    from fractalstat.exp04_fractal_scaling import run_fractal_scaling_test
    
    results = run_fractal_scaling_test(quick_mode=True)
    print(f"Fractal scaling validated: {results.is_fractal}")
"""

from .entities import ScaleTestConfig, ScaleTestResults, FractalScalingResults
from .experiment import run_fractal_scaling_test, run_scale_test, save_results

__all__ = [
    'ScaleTestConfig',
    'ScaleTestResults', 
    'FractalScalingResults',
    'run_fractal_scaling_test',
    'run_scale_test',
    'save_results'
]