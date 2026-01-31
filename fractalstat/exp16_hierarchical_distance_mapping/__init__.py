"""
EXP-16: Hierarchical Distance to Euclidean Distance Mapping

Tests whether hierarchical distance (discrete tree hops) maps to Euclidean distance
(continuous spatial distance) through fractal embedding strategies.

CORE HYPOTHESIS:
When fractal hierarchies are embedded in Euclidean space, hierarchical distance
d_h (tree hops) relates to Euclidean distance r through a power-law: r ∝ d_h^exponent.

This mapping explains why fractal gravity produces Newtonian-like inverse-square forces.

PHASES:
1. Build fractal embedding strategies (exponential, spherical, recursive)
2. Measure hierarchical vs Euclidean distances for embedded hierarchies
3. Find optimal power-law mapping relationship
4. Validate force scaling consistency between discrete and continuous

SUCCESS CRITERIA:
- Distance correlation > 0.95 for best embedding strategy
- Power-law exponent found (1 ≤ exponent ≤ 2)
- Force correlation > 0.90 between hierarchical and Euclidean
- Optimal embedding type identified

Classes:
- EmbeddedFractalHierarchy: Fractal hierarchy embedded in Euclidean space
- EmbeddingStrategy: Base class for embedding strategies
- ExponentialEmbedding: Exponential distance scaling embedding
- SphericalEmbedding: Spherical shell-based embedding
- RecursiveEmbedding: Recursive space partitioning embedding
- DistanceMappingAnalysis: Analysis of distance mapping relationships
- ForceScalingValidation: Validation of force scaling consistency
- EXP16_DistanceMappingResults: Complete experiment results

Usage:
    from fractalstat.exp16_hierarchical_distance_mapping import run_exp16_distance_mapping_experiment

    results = run_exp16_distance_mapping_experiment()
    print(f"Best embedding: {results.best_embedding_strategy}")
    print(f"Optimal exponent: {results.optimal_exponent}")
"""

__version__ = "1.0.0"
__author__ = "Tiny Walnut Games"
__description__ = "Hierarchical Distance to Euclidean Distance Mapping"

from .entities import (
    EmbeddedFractalHierarchy,
    EmbeddingStrategy,
    ExponentialEmbedding,
    SphericalEmbedding,
    RecursiveEmbedding,
    DistancePair,
    DistanceMappingAnalysis,
    ForceScalingValidation,
)
from .experiment import (
    EmbeddingTestResult,
    EXP16_DistanceMappingResults,
    create_embedding_strategies,
    measure_distances_in_embedding,
    analyze_distance_mapping,
    validate_force_scaling,
    test_embedding_strategy,
    run_exp16_distance_mapping_experiment,
    save_results,
)

__all__ = [
    # Core entities
    'EmbeddedFractalHierarchy',
    'EmbeddingStrategy',
    'ExponentialEmbedding',
    'SphericalEmbedding',
    'RecursiveEmbedding',
    'DistancePair',
    'DistanceMappingAnalysis',
    'ForceScalingValidation',

    # Experiment results and functions
    'EmbeddingTestResult',
    'EXP16_DistanceMappingResults',
    'create_embedding_strategies',
    'measure_distances_in_embedding',
    'analyze_distance_mapping',
    'validate_force_scaling',
    'test_embedding_strategy',
    'run_exp16_distance_mapping_experiment',
    'save_results',
]
