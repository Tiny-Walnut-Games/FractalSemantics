"""
EXP-11b: Dimensional Collision Stress Test

This module deliberately "dumbs down" the addressing system to test
the actual collision resistance provided by the dimensional structure itself,
independent of SHA-256's cryptographic guarantees.

The goal: Find out how "dumb" the system has to be before we see collisions.

Key Features:
- Tests collision resistance with fixed IDs and state
- Measures collision rates with limited coordinate ranges
- Tests minimal dimension counts (1, 2, 3 dimensions)
- Identifies when SHA-256 becomes the primary collision resistance mechanism
- Reveals whether dimensional structure provides meaningful differentiation

Main Classes:
- StressTestResult: Results for a single stress test configuration
- DimensionStressTestResult: Complete results from dimensional collision stress testing
- DimensionStressTest: Main experiment runner

Usage:
    from fractalstat.exp11b_dimension_stress_test import DimensionStressTest
    
    experiment = DimensionStressTest(sample_size=10000)
    results, success = experiment.run()
    print(f"First collisions appeared in: {results.key_findings[0]}")
    print(f"Max collision rate: {max(r.collision_rate for r in results.test_results):.4%}")
"""

from .entities import (
    StressTestResult,
    DimensionStressTestResult
)

from .experiment import (
    DimensionStressTest,
    save_results
)

__all__ = [
    # Data structures
    'StressTestResult',
    'DimensionStressTestResult',
    
    # Core classes
    'DimensionStressTest',
    
    # Utility functions
    'save_results',
]

__version__ = "1.0.0"
__author__ = "FractalStat Team"
