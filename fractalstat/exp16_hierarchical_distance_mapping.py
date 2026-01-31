"""
EXP-16: Hierarchical Distance to Euclidean Distance Mapping (Modular Version)

Tests whether hierarchical distance (discrete tree hops) maps to Euclidean distance
(continuous spatial distance) through fractal embedding strategies.

This is the modular version of EXP-16 that imports from the dedicated module.

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
"""

import sys
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Import from the modularized EXP-16
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp16_hierarchical_distance_mapping import (
    run_exp16_distance_mapping_experiment,
    save_results,
)

# Import configuration system
try:
    from fractalstat.config import ExperimentConfig
except ImportError:
    ExperimentConfig = None


def run_exp16_distance_mapping_experiment_modular(
    hierarchy_depth: int = 5,
    branching_factor: int = 3,
    scale_factors: List[float] = None,
    distance_samples: int = 1000
) -> Dict[str, any]:
    """
    Run EXP-16: Complete distance mapping experiment using modular components.

    Args:
        hierarchy_depth: Maximum depth of fractal hierarchies
        branching_factor: Branching factor for hierarchies
        scale_factors: Scale factors to test for embeddings
        distance_samples: Number of distance pairs to sample

    Returns:
        Complete experiment results
    """
    if scale_factors is None:
        scale_factors = [0.5, 1.0, 1.5, 2.0]

    start_time = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 80)
    print("EXP-16: HIERARCHICAL DISTANCE TO EUCLIDEAN DISTANCE MAPPING")
    print("=" * 80)
    print(f"Hierarchy depth: {hierarchy_depth}")
    print(f"Branching factor: {branching_factor}")
    print(f"Scale factors: {scale_factors}")
    print(f"Distance samples: {distance_samples}")
    print()

    # Create and run experiment using modular components
    results = run_exp16_distance_mapping_experiment(
        hierarchy_depth=hierarchy_depth,
        branching_factor=branching_factor,
        scale_factors=scale_factors,
        distance_samples=distance_samples
    )

    print("\n" + "=" * 70)
    print("CROSS-STRATEGY ANALYSIS")
    print("=" * 70)
    print(f"Best embedding strategy: {results.best_embedding_strategy}")
    print(f"Optimal power-law exponent: {results.optimal_exponent:.4f}")
    print(f"Distance mapping success: {'YES' if results.distance_mapping_success else 'NO'}")
    print(f"Force scaling consistent: {'YES' if results.force_scaling_consistent else 'NO'}")
    print(f"Experiment success: {'YES' if results.experiment_success else 'NO'}")
    print()

    return results


if __name__ == "__main__":
    # Load from config or use defaults
    hierarchy_depth = None
    branching_factor = None
    scale_factors = None
    distance_samples = None

    try:
        if ExperimentConfig:
            config = ExperimentConfig()
            hierarchy_depth = config.get("EXP-16", "hierarchy_depth", 5)
            branching_factor = config.get("EXP-16", "branching_factor", 3)
            scale_factors = config.get("EXP-16", "scale_factors", [0.5, 1.0, 1.5, 2.0])
            distance_samples = config.get("EXP-16", "distance_samples", 1000)
    except Exception:
        pass  # Use defaults

    # Check CLI args for quick/full modes
    if "--quick" in sys.argv:
        hierarchy_depth = 4
        branching_factor = 2
        scale_factors = [1.0]
        distance_samples = 100
    elif "--full" in sys.argv:
        # Use all available configurations
        pass

    try:
        results = run_exp16_distance_mapping_experiment_modular(
            hierarchy_depth=hierarchy_depth,
            branching_factor=branching_factor,
            scale_factors=scale_factors,
            distance_samples=distance_samples
        )
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("EXP-16 COMPLETE")
        print("=" * 80)

        status = "PASSED" if results.experiment_success else "FAILED"
        print(f"Status: {status}")
        print(f"Best embedding strategy: {results.best_embedding_strategy}")
        print(f"Optimal exponent: {results.optimal_exponent:.4f}")
        print(f"Output: {output_file}")
        print()

        if results.experiment_success:
            print("FUNDAMENTAL BREAKTHROUGH:")
            print("✓ Hierarchical distance maps to Euclidean distance")
            print("✓ Power-law relationship discovered: r ∝ d_h^exponent")
            print("✓ Force scaling consistent between discrete and continuous")
            print("✓ Fractal embedding explains inverse-square law emergence")
            print()
            print("The bridge between discrete fractal physics and continuous Newtonian physics is established!")
        else:
            print("Distance mapping not fully established.")
            print("Further investigation needed.")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
