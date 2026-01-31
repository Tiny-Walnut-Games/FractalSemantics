"""
EXP-11: Dimension Cardinality Analysis

This module explores the pros and cons of 7 dimensions vs. more or fewer dimensions.
It tests collision rates, retrieval performance, storage efficiency, and semantic
expressiveness across different dimension counts (3-10 dimensions).

Key Features:
- Tests collision rates vs. dimension count relationship
- Measures retrieval performance impact of dimension count
- Analyzes storage overhead per dimension
- Evaluates semantic disambiguation power
- Identifies diminishing returns beyond 7 dimensions
- Validates optimal dimension count for FractalStat addressing

Main Classes:
- DimensionTestResult: Results for a single dimension count test
- DimensionCardinalityResult: Complete results from dimension cardinality analysis
- EXP11_DimensionCardinality: Main experiment runner

Usage:
    from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality
    
    experiment = EXP11_DimensionCardinality(sample_size=1000)
    results, success = experiment.run()
    print(f"Optimal dimensions: {results.optimal_dimension_count}")
    print(f"Collision rate: {results.optimal_collision_rate:.4%}")
"""

from .entities import (
    DimensionTestResult,
    DimensionCardinalityResult
)

from .experiment import (
    EXP11_DimensionCardinality,
    save_results
)

__all__ = [
    # Data structures
    'DimensionTestResult',
    'DimensionCardinalityResult',
    
    # Core classes
    'EXP11_DimensionCardinality',
    
    # Utility functions
    'save_results',
]

__version__ = "1.0.0"
__author__ = "FractalStat Team"
