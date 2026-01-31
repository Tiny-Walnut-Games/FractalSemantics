"""
EXP-09: FractalStat Performance Under Memory Pressure

This module implements comprehensive memory pressure testing for FractalStat systems.
It tests system resilience and performance under constrained memory conditions,
demonstrating real-world viability through stress testing and optimization.

Key Features:
- Memory pressure testing with controlled allocation patterns
- Performance degradation analysis under load
- Garbage collection effectiveness measurement
- Memory optimization strategy testing
- System stability and breaking point identification

Main Classes:
- MemoryPressureMetrics: Metrics collected during memory pressure testing
- StressTestPhase: Represents a phase in the memory stress testing
- MemoryOptimization: Memory optimization strategy applied during testing
- MemoryPressureResults: Results from the memory pressure test
- MemoryPressureTester: System for testing FractalStat performance under memory pressure
- MemoryPressureExperiment: Main experiment runner

Usage:
    from fractalstat.exp09_memory_pressure import MemoryPressureExperiment
    
    experiment = MemoryPressureExperiment(max_memory_target_mb=1000)
    results = experiment.run()
    print(f"Experiment status: {results.status}")
    print(f"Peak memory usage: {results.peak_memory_usage_mb:.1f}MB")
"""

from .entities import (
    MemoryPressureMetrics,
    StressTestPhase,
    MemoryOptimization,
    MemoryPressureResults
)

from .experiment import (
    MemoryPressureTester,
    MemoryPressureExperiment
)

__all__ = [
    # Data structures
    'MemoryPressureMetrics',
    'StressTestPhase',
    'MemoryOptimization',
    'MemoryPressureResults',
    
    # Core classes
    'MemoryPressureTester',
    'MemoryPressureExperiment',
]

__version__ = "1.0.0"
__author__ = "FractalStat Team"