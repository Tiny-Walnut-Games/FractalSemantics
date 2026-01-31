"""
EXP-10: Multi-Dimensional Query Optimization

This module implements comprehensive multi-dimensional query optimization for FractalStat systems.
It demonstrates FractalStat's unique querying capabilities across all dimensions, showcasing
practical value proposition and differentiation from traditional systems.

Key Features:
- Multi-dimensional query execution across all FractalStat dimensions
- Query optimization strategies (indexing, caching, pruning, parallelization)
- Performance analysis and benchmarking
- Real-world use case validation
- Scalability testing and analysis

Main Classes:
- QueryPattern: Definition of multi-dimensional query patterns
- QueryResult: Results from executing multi-dimensional queries
- QueryOptimizer: Query optimization strategies
- MultiDimensionalQueryResults: Complete experiment results
- MultiDimensionalQueryEngine: Query engine for FractalStat queries
- MultiDimensionalQueryExperiment: Main experiment runner

Usage:
    from fractalstat.exp10_multidimensional_query import MultiDimensionalQueryExperiment
    
    experiment = MultiDimensionalQueryExperiment(dataset_size=10000)
    results = experiment.run()
    print(f"Average query time: {results.avg_query_time_ms:.2f}ms")
    print(f"Query precision: {results.avg_precision:.3f}")
"""

from .entities import (
    QueryPattern,
    QueryResult,
    QueryOptimizer,
    MultiDimensionalQueryResults
)

from .experiment import (
    MultiDimensionalQueryEngine,
    MultiDimensionalQueryExperiment
)

__all__ = [
    # Data structures
    'QueryPattern',
    'QueryResult',
    'QueryOptimizer',
    'MultiDimensionalQueryResults',
    
    # Core classes
    'MultiDimensionalQueryEngine',
    'MultiDimensionalQueryExperiment',
]

__version__ = "1.0.0"
__author__ = "FractalStat Team"