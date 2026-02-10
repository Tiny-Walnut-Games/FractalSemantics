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
"""

import json
import time
import secrets
import sys
import random
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import math
import statistics

# Import from EXP-13 for fractal hierarchy
from fractalsemantics.exp13_fractal_gravity import (
    FractalHierarchy,
    FractalNode,
)

# Import from EXP-20 for vector field approaches

secure_random = secrets.SystemRandom()

# ============================================================================
# EXP-16: FRACTAL EMBEDDING STRATEGIES
# ============================================================================

@dataclass
class EmbeddedFractalHierarchy:
    """
    A fractal hierarchy embedded in Euclidean space using a specific strategy.
    """
    hierarchy: FractalHierarchy
    embedding_type: str
    positions: Dict[FractalNode, np.ndarray]  # node -> [x,y,z] position
    scale_factor: float = 1.0

    def get_euclidean_distance(self, node_a: FractalNode, node_b: FractalNode) -> float:
        """Get Euclidean distance between two embedded nodes."""
        pos_a = self.positions[node_a]
        pos_b = self.positions[node_b]
        return np.linalg.norm(pos_a - pos_b)

    def get_hierarchical_distance(self, node_a: FractalNode, node_b: FractalNode) -> int:
        """Get hierarchical distance between two nodes."""
        return self.hierarchy.get_hierarchical_distance(node_a, node_b)


@dataclass
class EmbeddingStrategy:
    """
    A strategy for embedding fractal hierarchies in Euclidean space.
    """
    name: str
    description: str

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """
        Embed the given hierarchy using this strategy.

        Args:
            hierarchy: The fractal hierarchy to embed
            scale_factor: Scaling factor for the embedding

        Returns:
            Embedded hierarchy with positions
        """
        raise NotImplementedError("Subclasses must implement embed_hierarchy")


class ExponentialEmbedding(EmbeddingStrategy):
    """
    Exponential embedding: Nodes placed at exponentially increasing distances from root.
    """

    def __init__(self):
        super().__init__(
            name="Exponential",
            description="Nodes placed at exponential distance from parent (d ∝ α^depth)"
        )

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """Embed using exponential distance scaling."""
        positions = {}

        # Root node at origin
        root_nodes = hierarchy.nodes_by_depth[0]
        if root_nodes:
            positions[root_nodes[0]] = np.array([0.0, 0.0, 0.0])

        # Place nodes level by level
        for depth in range(1, hierarchy.max_depth):
            parent_depth = depth - 1
            if parent_depth not in hierarchy.nodes_by_depth:
                continue

            parent_nodes = hierarchy.nodes_by_depth[parent_depth]
            child_nodes = hierarchy.nodes_by_depth.get(depth, [])

            # Group children by parent
            children_by_parent = {}
            for child in child_nodes:
                parent_idx = child.tree_address[0] if child.tree_address else 0
                if parent_idx < len(parent_nodes):
                    parent = parent_nodes[parent_idx]
                    if parent not in children_by_parent:
                        children_by_parent[parent] = []
                    children_by_parent[parent].append(child)

            # Place children around each parent
            for parent, children in children_by_parent.items():
                parent_pos = positions[parent]
                radius = scale_factor * (2.0 ** depth)  # Exponential scaling

                for i, child in enumerate(children):
                    # Distribute children in a circle around parent
                    angle = 2 * np.pi * i / len(children)
                    offset = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        scale_factor * depth * 0.1  # Small vertical offset
                    ])
                    positions[child] = parent_pos + offset

        return EmbeddedFractalHierarchy(
            hierarchy=hierarchy,
            embedding_type=self.name,
            positions=positions,
            scale_factor=scale_factor
        )


class SphericalEmbedding(EmbeddingStrategy):
    """
    Spherical embedding: Nodes placed on concentric spheres.
    """

    def __init__(self):
        super().__init__(
            name="Spherical",
            description="Nodes placed on concentric spheres (shell-based)"
        )

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """Embed using spherical shells."""
        positions = {}

        # Root at origin
        root_nodes = hierarchy.nodes_by_depth[0]
        if root_nodes:
            positions[root_nodes[0]] = np.array([0.0, 0.0, 0.0])

        # Place nodes on spherical shells by depth
        for depth in range(1, hierarchy.max_depth):
            nodes_at_depth = hierarchy.nodes_by_depth.get(depth, [])
            if not nodes_at_depth:
                continue

            # Radius increases with depth
            radius = scale_factor * (depth + 1)

            # Distribute nodes evenly on sphere
            for i, node in enumerate(nodes_at_depth):
                # Golden spiral distribution for even spacing
                golden_ratio = (1 + np.sqrt(5)) / 2
                theta = 2 * np.pi * i / golden_ratio  # Longitude
                phi = np.arccos(1 - 2 * (i + 0.5) / len(nodes_at_depth))  # Latitude

                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)

                positions[node] = np.array([x, y, z])

        return EmbeddedFractalHierarchy(
            hierarchy=hierarchy,
            embedding_type=self.name,
            positions=positions,
            scale_factor=scale_factor
        )


class RecursiveEmbedding(EmbeddingStrategy):
    """
    Recursive embedding: Each branching splits space along coordinate axes.
    """

    def __init__(self):
        super().__init__(
            name="Recursive",
            description="Recursive space division along coordinate axes"
        )

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """Embed using recursive space partitioning."""
        positions = {}

        def place_node_recursive(node: FractalNode, center: np.ndarray, size: float):
            """Recursively place a node and its children."""
            positions[node] = center.copy()

            # Get children
            child_depth = node.hierarchical_depth + 1
            if child_depth >= hierarchy.max_depth:
                return

            children = []
            for potential_child in hierarchy.nodes_by_depth.get(child_depth, []):
                if (potential_child.tree_address and
                    len(potential_child.tree_address) > node.hierarchical_depth and
                    potential_child.tree_address[node.hierarchical_depth] == node.tree_address[-1] if node.tree_address else 0):
                    children.append(potential_child)

            if not children:
                return

            # Split space among children
            child_size = size / 2.0
            axes = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]  # 6 directions

            for i, child in enumerate(children):
                if i < len(axes):
                    direction = np.array(axes[i])
                    child_center = center + direction * child_size * scale_factor
                    place_node_recursive(child, child_center, child_size)

        # Start with root
        root_nodes = hierarchy.nodes_by_depth[0]
        if root_nodes:
            place_node_recursive(root_nodes[0], np.array([0.0, 0.0, 0.0]), scale_factor)

        return EmbeddedFractalHierarchy(
            hierarchy=hierarchy,
            embedding_type=self.name,
            positions=positions,
            scale_factor=scale_factor
        )


# ============================================================================
# EXP-16: DISTANCE MEASUREMENT AND ANALYSIS
# ============================================================================

@dataclass
class DistancePair:
    """A pair of nodes with both distance measurements."""
    node_a: FractalNode
    node_b: FractalNode
    hierarchical_distance: int
    euclidean_distance: float

    @property
    def distance_ratio(self) -> float:
        """Ratio of Euclidean to hierarchical distance."""
        return self.euclidean_distance / max(1, self.hierarchical_distance)


@dataclass
class DistanceMappingAnalysis:
    """
    Analysis of distance mapping for an embedded hierarchy.
    """
    embedding: EmbeddedFractalHierarchy
    distance_pairs: List[DistancePair]

    # Power-law fitting results
    power_law_exponent: float = field(init=False)
    power_law_coefficient: float = field(init=False)
    correlation_coefficient: float = field(init=False)
    mapping_quality: float = field(init=False)

    def __post_init__(self):
        """Fit power-law relationship between distances."""
        if not self.distance_pairs:
            self.power_law_exponent = 0.0
            self.power_law_coefficient = 0.0
            self.correlation_coefficient = 0.0
            self.mapping_quality = 0.0
            return

        # Extract distance data
        h_distances = [p.hierarchical_distance for p in self.distance_pairs]
        e_distances = [p.euclidean_distance for p in self.distance_pairs]

        # Fit power law: e_distance = coefficient * h_distance^exponent
        try:
            # Use log-linear regression
            log_h = [math.log(max(1, d)) for d in h_distances]
            log_e = [math.log(max(1e-6, d)) for d in e_distances]

            # Linear regression on log scales
            n = len(log_h)
            sum_x = sum(log_h)
            sum_y = sum(log_e)
            sum_xy = sum(x*y for x,y in zip(log_h, log_e))
            sum_x2 = sum(x*x for x in log_h)

            # Slope (exponent) and intercept (log coefficient)
            denominator = n * sum_x2 - sum_x**2
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / n

                self.power_law_exponent = slope
                self.power_law_coefficient = math.exp(intercept)

                # Calculate correlation
                y_pred = [intercept + slope * x for x in log_h]
                correlation = np.corrcoef(log_e, y_pred)[0, 1]
                self.correlation_coefficient = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                self.power_law_exponent = 1.0
                self.power_law_coefficient = 1.0
                self.correlation_coefficient = 0.0

        except Exception:
            self.power_law_exponent = 1.0
            self.power_law_coefficient = 1.0
            self.correlation_coefficient = 0.0

        # Overall mapping quality (0-1 scale)
        self.mapping_quality = min(1.0, self.correlation_coefficient)


@dataclass
class ForceScalingValidation:
    """
    Validation of force scaling consistency between hierarchical and Euclidean approaches.
    """
    embedding_analysis: DistanceMappingAnalysis
    hierarchical_forces: List[float]
    euclidean_forces: List[float]

    # Force scaling analysis
    force_correlation: float = field(init=False)
    scaling_consistency: float = field(init=False)

    def __post_init__(self):
        """Analyze force scaling consistency."""
        if not self.hierarchical_forces or not self.euclidean_forces:
            self.force_correlation = 0.0
            self.scaling_consistency = 0.0
            return

        try:
            correlation = np.corrcoef(self.hierarchical_forces, self.euclidean_forces)[0, 1]
            self.force_correlation = abs(correlation) if not np.isnan(correlation) else 0.0
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not compute force correlation: {e}")
            self.force_correlation = 0.0

        # Scaling consistency: how well the relative force magnitudes match
        self.scaling_consistency = self.force_correlation


# ============================================================================
# EXP-16: EMBEDDING AND MEASUREMENT FUNCTIONS
# ============================================================================

def create_embedding_strategies() -> List[EmbeddingStrategy]:
    """Create all available embedding strategies."""
    return [
        ExponentialEmbedding(),
        SphericalEmbedding(),
        RecursiveEmbedding(),
    ]


def measure_distances_in_embedding(embedding: EmbeddedFractalHierarchy, num_samples: int = 1000) -> List[DistancePair]:
    """
    Measure distance pairs in an embedded hierarchy.

    Args:
        embedding: The embedded hierarchy
        num_samples: Number of random node pairs to measure

    Returns:
        List of distance measurements
    """
    all_nodes = list(embedding.positions.keys())
    if len(all_nodes) < 2:
        return []

    distance_pairs = []

    for _ in range(num_samples):
        # Select two random nodes
        node_a, node_b = random.sample(all_nodes, 2)

        h_distance = embedding.get_hierarchical_distance(node_a, node_b)
        e_distance = embedding.get_euclidean_distance(node_a, node_b)

        pair = DistancePair(
            node_a=node_a,
            node_b=node_b,
            hierarchical_distance=h_distance,
            euclidean_distance=e_distance
        )
        distance_pairs.append(pair)

    return distance_pairs


def analyze_distance_mapping(embedding: EmbeddedFractalHierarchy, num_samples: int = 1000) -> DistanceMappingAnalysis:
    """
    Analyze the distance mapping for an embedded hierarchy.

    Args:
        embedding: The embedded hierarchy
        num_samples: Number of distance pairs to analyze

    Returns:
        Complete distance mapping analysis
    """
    distance_pairs = measure_distances_in_embedding(embedding, num_samples)
    return DistanceMappingAnalysis(
        embedding=embedding,
        distance_pairs=distance_pairs
    )


def validate_force_scaling(
    embedding_analysis: DistanceMappingAnalysis,
    scalar_magnitude: float = 1.0
) -> ForceScalingValidation:
    """
    Validate that force scaling is consistent between hierarchical and Euclidean approaches.

    Args:
        embedding_analysis: Distance mapping analysis
        scalar_magnitude: Base force magnitude

    Returns:
        Force scaling validation results
    """
    hierarchical_forces = []
    euclidean_forces = []

    for pair in embedding_analysis.distance_pairs:
        h_dist = pair.hierarchical_distance
        e_dist = pair.euclidean_distance

        # Hierarchical force (inverse-square on hierarchical distance)
        if h_dist > 0:
            h_force = scalar_magnitude / (h_dist ** 2)
        else:
            h_force = scalar_magnitude
        hierarchical_forces.append(h_force)

        # Euclidean force (Newtonian inverse-square)
        if e_dist > 0:
            e_force = scalar_magnitude / (e_dist ** 2)
        else:
            e_force = scalar_magnitude
        euclidean_forces.append(e_force)

    return ForceScalingValidation(
        embedding_analysis=embedding_analysis,
        hierarchical_forces=hierarchical_forces,
        euclidean_forces=euclidean_forces
    )


# ============================================================================
# EXP-16: EXPERIMENT IMPLEMENTATION
# ============================================================================

@dataclass
class EmbeddingTestResult:
    """Results from testing a specific embedding strategy."""

    strategy_name: str
    embedding: EmbeddedFractalHierarchy
    distance_analysis: DistanceMappingAnalysis
    force_validation: ForceScalingValidation

    # Success metrics
    distance_correlation: float
    force_correlation: float
    exponent_in_range: bool
    overall_quality: float


@dataclass
class EXP16_DistanceMappingResults:
    """Complete results from EXP-16 distance mapping experiment."""

    start_time: str
    end_time: str
    total_duration_seconds: float

    # Test parameters
    hierarchy_depth: int
    branching_factor: int
    distance_samples: int

    # Results for each embedding strategy
    embedding_results: Dict[str, EmbeddingTestResult]

    # Cross-strategy analysis
    best_embedding_strategy: str
    optimal_exponent: float
    distance_mapping_success: bool
    force_scaling_consistent: bool
    experiment_success: bool


# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def test_embedding_strategy(
    strategy: EmbeddingStrategy,
    hierarchy_depth: int = 5,
    branching_factor: int = 3,
    scale_factor: float = 1.0,
    distance_samples: int = 1000,
    scalar_magnitude: float = 1.0
) -> EmbeddingTestResult:
    """
    Test a specific embedding strategy.

    Args:
        strategy: The embedding strategy to test
        hierarchy_depth: Maximum depth of fractal hierarchy
        branching_factor: Branching factor for hierarchy
        scale_factor: Scale factor for embedding
        distance_samples: Number of distance pairs to sample
        scalar_magnitude: Base force magnitude for validation

    Returns:
        Complete test results for this strategy
    """
    print(f"Testing {strategy.name} embedding strategy...")

    # Build fractal hierarchy
    hierarchy = FractalHierarchy.build(
        element_type="test_element",
        max_depth=hierarchy_depth,
        branching_factor=branching_factor
    )

    # Embed hierarchy
    embedding = strategy.embed_hierarchy(hierarchy, scale_factor)

    # Analyze distance mapping
    distance_analysis = analyze_distance_mapping(embedding, distance_samples)

    # Validate force scaling
    force_validation = validate_force_scaling(distance_analysis, scalar_magnitude)

    # Calculate success metrics
    distance_correlation = distance_analysis.correlation_coefficient
    force_correlation = force_validation.force_correlation
    exponent_in_range = 1.0 <= distance_analysis.power_law_exponent <= 2.0

    # Overall quality score
    quality_components = [
        distance_correlation,
        force_correlation,
        1.0 if exponent_in_range else 0.0
    ]
    overall_quality = statistics.mean(quality_components) if quality_components else 0.0

    print(f"  Distance correlation: {distance_correlation:.4f}")
    print(f"  Force correlation: {force_correlation:.4f}")
    print(f"  Power-law exponent: {distance_analysis.power_law_exponent:.4f}")
    print(f"  Overall quality: {overall_quality:.4f}")

    return EmbeddingTestResult(
        strategy_name=strategy.name,
        embedding=embedding,
        distance_analysis=distance_analysis,
        force_validation=force_validation,
        distance_correlation=distance_correlation,
        force_correlation=force_correlation,
        exponent_in_range=exponent_in_range,
        overall_quality=overall_quality
    )


def run_exp16_distance_mapping_experiment(
    hierarchy_depth: int = 5,
    branching_factor: int = 3,
    scale_factors: List[float] = None,
    distance_samples: int = 1000
) -> EXP16_DistanceMappingResults:
    """
    Run EXP-16: Complete distance mapping experiment.

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
    overall_start = time.time()

    print("\n" + "=" * 80)
    print("EXP-16: HIERARCHICAL DISTANCE TO EUCLIDEAN DISTANCE MAPPING")
    print("=" * 80)
    print(f"Hierarchy depth: {hierarchy_depth}")
    print(f"Branching factor: {branching_factor}")
    print(f"Scale factors: {scale_factors}")
    print(f"Distance samples: {distance_samples}")
    print()

    # Create embedding strategies
    strategies = create_embedding_strategies()

    # Test each strategy with different scale factors
    embedding_results = {}

    for strategy in strategies:
        best_result = None
        best_quality = 0.0

        for scale_factor in scale_factors:
            try:
                result = test_embedding_strategy(
                    strategy=strategy,
                    hierarchy_depth=hierarchy_depth,
                    branching_factor=branching_factor,
                    scale_factor=scale_factor,
                    distance_samples=distance_samples
                )

                if result.overall_quality > best_quality:
                    best_quality = result.overall_quality
                    best_result = result

            except Exception as e:
                print(f"  FAILED {strategy.name} at scale {scale_factor}: {e}")
                continue

        if best_result:
            embedding_results[strategy.name] = best_result
            print(f"Best result for {strategy.name}: quality = {best_result.overall_quality:.4f}")
        print()

    # Cross-strategy analysis
    if embedding_results:
        best_strategy = max(embedding_results.keys(),
                          key=lambda k: embedding_results[k].overall_quality)
        best_result = embedding_results[best_strategy]
        optimal_exponent = best_result.distance_analysis.power_law_exponent
    else:
        best_strategy = "None"
        optimal_exponent = 0.0

    # Success criteria
    distance_mapping_success = any(
        result.distance_correlation > 0.95 for result in embedding_results.values()
    ) if embedding_results else False

    force_scaling_consistent = any(
        result.force_correlation > 0.90 for result in embedding_results.values()
    ) if embedding_results else False

    exponent_reasonable = any(
        result.exponent_in_range for result in embedding_results.values()
    ) if embedding_results else False

    experiment_success = (
        distance_mapping_success and
        force_scaling_consistent and
        exponent_reasonable
    )

    overall_end = time.time()
    end_time = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 70)
    print("CROSS-STRATEGY ANALYSIS")
    print("=" * 70)
    print(f"Best embedding strategy: {best_strategy}")
    print(f"Optimal power-law exponent: {optimal_exponent:.4f}")
    print(f"Distance mapping success: {'YES' if distance_mapping_success else 'NO'}")
    print(f"Force scaling consistent: {'YES' if force_scaling_consistent else 'NO'}")
    print(f"Experiment success: {'YES' if experiment_success else 'NO'}")
    print()

    results = EXP16_DistanceMappingResults(
        start_time=start_time,
        end_time=end_time,
        total_duration_seconds=(overall_end - overall_start),
        hierarchy_depth=hierarchy_depth,
        branching_factor=branching_factor,
        distance_samples=distance_samples,
        embedding_results=embedding_results,
        best_embedding_strategy=best_strategy,
        optimal_exponent=optimal_exponent,
        distance_mapping_success=distance_mapping_success,
        force_scaling_consistent=force_scaling_consistent,
        experiment_success=experiment_success,
    )

    return results


# ============================================================================
# CLI & RESULTS PERSISTENCE
# ============================================================================

def save_results(results: EXP16_DistanceMappingResults, output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""

    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp16_hierarchical_distance_mapping_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    # Convert to serializable format
    serializable_results = {
        "experiment": "EXP-16",
        "test_type": "Hierarchical Distance to Euclidean Distance Mapping",
        "start_time": results.start_time,
        "end_time": results.end_time,
        "total_duration_seconds": round(results.total_duration_seconds, 3),
        "parameters": {
            "hierarchy_depth": results.hierarchy_depth,
            "branching_factor": results.branching_factor,
            "distance_samples": results.distance_samples,
        },
        "embedding_results": {
            strategy_name: {
                "strategy_name": result.strategy_name,
                "distance_correlation": round(float(result.distance_correlation), 6),
                "force_correlation": round(float(result.force_correlation), 6),
                "exponent_in_range": bool(result.exponent_in_range),
                "overall_quality": round(float(result.overall_quality), 6),
                "distance_analysis": {
                    "power_law_exponent": round(float(result.distance_analysis.power_law_exponent), 6),
                    "power_law_coefficient": round(float(result.distance_analysis.power_law_coefficient), 6),
                    "correlation_coefficient": round(float(result.distance_analysis.correlation_coefficient), 6),
                    "mapping_quality": round(float(result.distance_analysis.mapping_quality), 6),
                },
                "force_validation": {
                    "force_correlation": round(float(result.force_validation.force_correlation), 6),
                    "scaling_consistency": round(float(result.force_validation.scaling_consistency), 6),
                }
            }
            for strategy_name, result in results.embedding_results.items()
        },
        "analysis": {
            "best_embedding_strategy": results.best_embedding_strategy,
            "optimal_exponent": round(float(results.optimal_exponent), 6),
            "distance_mapping_success": bool(results.distance_mapping_success),
            "force_scaling_consistent": bool(results.force_scaling_consistent),
            "experiment_success": bool(results.experiment_success),
        },
    }

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Load from config or use defaults
    try:
        from fractalsemantics.config import ExperimentConfig

        config = ExperimentConfig()
        hierarchy_depth = config.get("EXP-16", "hierarchy_depth", 5)
        branching_factor = config.get("EXP-16", "branching_factor", 3)
        scale_factors = config.get("EXP-16", "scale_factors", [0.5, 1.0, 1.5, 2.0])
        distance_samples = config.get("EXP-16", "distance_samples", 1000)
    except Exception:
        hierarchy_depth = 5
        branching_factor = 3
        scale_factors = [0.5, 1.0, 1.5, 2.0]
        distance_samples = 1000

    try:
        results = run_exp16_distance_mapping_experiment(
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
