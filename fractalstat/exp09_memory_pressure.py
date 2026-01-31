"""
EXP-09: FractalStat Performance Under Memory Pressure

Tests system resilience and performance under constrained memory conditions,
demonstrating real-world viability through stress testing and optimization.

Hypothesis:
FractalStat maintains performance and stability under memory pressure through:
- Efficient memory usage optimization strategies
- Graceful performance degradation patterns
- Effective garbage collection and memory management
- Scalability limits with clear breaking points

Methodology:
1. Generate large-scale bit-chain datasets with memory constraints
2. Apply memory pressure through controlled allocation and retention
3. Measure performance degradation patterns under load
4. Test garbage collection effectiveness and memory management
5. Identify scalability limits and breaking points
6. Validate optimization strategies under stress

Success Criteria:
- Performance degrades gracefully (no sudden drops)
- Memory usage remains bounded under load
- Garbage collection maintains system stability
- Breaking points are predictable and documented
- Optimization strategies improve resilience by >30%

This module has been modularized for better maintainability and reusability.
For the complete implementation, see the fractalstat.exp09_memory_pressure module.
"""

from fractalstat.exp09_memory_pressure import (
    MemoryPressureExperiment,
    MemoryPressureTester,
    MemoryPressureResults,
    MemoryPressureMetrics,
    StressTestPhase,
    MemoryOptimization,
    save_results
)

__all__ = [
    'MemoryPressureExperiment',
    'MemoryPressureTester', 
    'MemoryPressureResults',
    'MemoryPressureMetrics',
    'StressTestPhase',
    'MemoryOptimization',
    'save_results'
]


def main():
    """Main entry point for EXP-09."""
    import sys
    
    # Load from config or use defaults
    max_memory_target_mb = 1000
    
    try:
        from fractalstat.config import ExperimentConfig
        config = ExperimentConfig()
        max_memory_target_mb = config.get("EXP-09", "max_memory_target_mb", 1000)
    except Exception:
        pass
    
    # Override based on command line
    if "--quick" in sys.argv:
        max_memory_target_mb = 200
    elif "--full" in sys.argv:
        max_memory_target_mb = 2000
    
    try:
        experiment = MemoryPressureExperiment(max_memory_target_mb=max_memory_target_mb)
        results = experiment.run()
        
        output_file = save_results(results)
        
        print("\n" + "=" * 80)
        print("EXP-09 COMPLETE")
        print("=" * 80)
        print(f"Status: {results.status}")
        print(f"Peak Memory Usage: {results.peak_memory_usage_mb:.1f}MB")
        print(f"Performance Degradation: {results.degradation_ratio:.1f}x")
        print(f"Stability Score: {results.stability_score:.3f}")
        print(f"Optimization Improvement: {results.optimization_improvement:.1%}")
        print(f"Graceful Degradation: {'Yes' if results.graceful_degradation else 'No'}")
        print(f"Results: {output_file}")
        print()
        
        return results.status == "PASS"
        
    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
