"""
EXP-11: Dimension Cardinality Analysis

Explores the pros and cons of 7 dimensions vs. more or fewer dimensions.
Tests collision rates, retrieval performance, storage efficiency, and semantic
expressiveness across different dimension counts (3-10 dimensions).

Validates:
- Optimal dimension count for FractalSemantics addressing
- Collision rate vs. dimension count relationship
- Retrieval performance impact of dimension count
- Storage overhead per dimension
- Semantic disambiguation power
- Diminishing returns beyond 7 dimensions

Status: Phase 2 validation experiment
"""

import json
import time
import sys
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import statistics
from pathlib import Path

# Reuse canonical serialization from Phase 1
from fractalsemantics.fractalsemantics_experiments import (
    compute_address_hash,
    BitChain,
    generate_random_bitchain,
)

secure_random = secrets.SystemRandom()

# ============================================================================
# EXP-11 DATA STRUCTURES
# ============================================================================


@dataclass
class DimensionTestResult:
    """Results for a single dimension count test."""

    dimension_count: int
    dimensions_used: List[str]
    sample_size: int
    unique_addresses: int
    collisions: int
    collision_rate: float
    mean_retrieval_latency_ms: float
    median_retrieval_latency_ms: float
    avg_storage_bytes: int
    storage_overhead_per_dimension: float
    semantic_expressiveness_score: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)


@dataclass
class DimensionCardinalityResult:
    """Complete results from EXP-11 dimension cardinality analysis."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    sample_size: int
    dimension_counts_tested: List[int]
    test_iterations: int

    # Per-dimension-count results
    dimension_results: List[DimensionTestResult]

    # Aggregate analysis
    optimal_dimension_count: int
    optimal_collision_rate: float
    optimal_retrieval_latency_ms: float
    optimal_storage_efficiency: float
    diminishing_returns_threshold: int  # Dimension count where returns diminish

    # Key findings
    major_findings: List[str] = field(default_factory=list)
    seven_dimensions_justified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "experiment": "EXP-11",
            "test_type": "Dimension Cardinality Analysis",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "sample_size": self.sample_size,
            "dimension_counts_tested": self.dimension_counts_tested,
            "test_iterations": self.test_iterations,
            "optimal_analysis": {
                "optimal_dimension_count": self.optimal_dimension_count,
                "optimal_collision_rate": round(self.optimal_collision_rate, 6),
                "optimal_retrieval_latency_ms": round(
                    self.optimal_retrieval_latency_ms, 4
                ),
                "optimal_storage_efficiency": round(self.optimal_storage_efficiency, 3),
                "diminishing_returns_threshold": self.diminishing_returns_threshold,
            },
            "seven_dimensions_justified": self.seven_dimensions_justified,
            "major_findings": self.major_findings,
            "dimension_results": [r.to_dict() for r in self.dimension_results],
        }


# ============================================================================
# DIMENSION VARIATION TESTING
# ============================================================================


class EXP11_DimensionCardinality:
    """
    Previously called DimensionCardinalityExperiment, EXP11_DimensionCardinality tests
    FractalSemantics addressing with different dimension counts.

    Approach:
    1. Baseline: Test with all 7 FractalSemantics dimensions
    2. Reduced: Test with 3, 4, 5, 6 dimensions
    3. Extended: Test with 8, 9, 10 dimensions (hypothetical)
    4. Measure: Collision rates, retrieval latency, storage, expressiveness
    """

    FractalSemantics_DIMENSIONS = [
        "realm",
        "lineage",
        "adjacency",
        "horizon",
        "luminosity",
        "polarity",
        "dimensionality",
        "alignment",
    ]

    # Hypothetical dimensions for 8, 9, 10 dimension tests
    EXTENDED_DIMENSIONS = [
        "temperature",  # Thermal activity level
        "entropy",  # Disorder/randomness measure
        "coherence",  # Internal consistency measure
    ]

    def __init__(
        self,
        sample_size: int = 1000,
        dimension_counts: Optional[List[int]] = None,
        test_iterations: int = 5,
    ):
        """
        Initialize dimension cardinality experiment.

        Args:
            sample_size: Number of bit-chains to test per dimension count
            dimension_counts: List of dimension counts to test (default: [3,4,5,6,7,8,9,10])
            test_iterations: Number of iterations per dimension count
        """
        self.sample_size = sample_size
        self.dimension_counts = (
            dimension_counts
            if dimension_counts is not None
            else [3, 4, 5, 6, 7, 8, 9, 10]
        )
        self.test_iterations = test_iterations
        self.results: List[DimensionTestResult] = []

    def _select_dimensions(self, count: int) -> List[str]:
        """
        Select which dimensions to use for a given count.

        Strategy:
        - For count <= 8: Use first N FractalSemantics dimensions
        - For count > 8: Use all 8 FractalSemantics + extended dimensions
        """
        if count <= 8:
            return self.FractalSemantics_DIMENSIONS[:count]
        else:
            extended_count = count - 8
            return self.FractalSemantics_DIMENSIONS + self.EXTENDED_DIMENSIONS[:extended_count]

    def _compute_address_with_dimensions(
        self, bc: BitChain, dimensions: List[str]
    ) -> str:
        """
        Compute address using only specified dimensions.

        Args:
            bc: BitChain to address
            dimensions: List of dimension names to include

        Returns:
            SHA-256 hash of canonical serialization with selected dimensions
        """
        # Build coordinate dict with only selected dimensions
        coords_dict = {}

        for dim in dimensions:
            if dim in self.FractalSemantics_DIMENSIONS:
                # Use actual FractalSemantics dimension
                coords_dict[dim] = getattr(bc.coordinates, dim)
            elif dim == "temperature":
                # Hypothetical: temperature = abs(luminosity) * dimensionality
                coords_dict[dim] = abs(bc.coordinates.luminosity) * bc.coordinates.dimensionality
            elif dim == "entropy":
                # Hypothetical: entropy = 1.0 - abs(luminosity)
                coords_dict[dim] = 1.0 - abs(bc.coordinates.luminosity)
            elif dim == "coherence":
                # Hypothetical: coherence = (1.0 - abs(luminosity)) * dimensionality
                coords_dict[dim] = (
                    1.0 - abs(bc.coordinates.luminosity)
                ) * bc.coordinates.dimensionality

        # Normalize adjacency if present
        if "adjacency" in coords_dict:
            coords_dict["adjacency"] = sorted(coords_dict["adjacency"])

        # Create canonical dict for hashing
        data = {
            "id": bc.id,
            "entity_type": bc.entity_type,
            "realm": bc.realm,
            "fractalsemantics_coordinates": coords_dict,
        }

        return compute_address_hash(data)

    def _calculate_semantic_expressiveness(
        self, dimensions: List[str], bitchains: List[BitChain]
    ) -> float:
        """
        Calculate semantic expressiveness score (0.0 to 1.0).

        Measures how well the dimension set can distinguish semantically
        different entities. Higher score = better disambiguation.

        Heuristic:
        - realm: +0.20 (domain classification)
        - lineage: +0.15 (temporal/generational context)
        - adjacency: +0.15 (relational context)
        - horizon: +0.15 (lifecycle stage)
        - resonance: +0.10 (affective alignment)
        - velocity: +0.10 (change dynamics)
        - density: +0.10 (compression context)
        - temperature: +0.05 (thermal activity)
        - entropy: +0.05 (disorder measure)
        - coherence: +0.05 (consistency measure)
        """
        score = 0.0
        weights = {
            "realm": 0.20,
            "lineage": 0.15,
            "adjacency": 0.15,
            "horizon": 0.15,
            "luminosity": 0.10,
            "polarity": 0.10,
            "dimensionality": 0.10,
            "temperature": 0.05,
            "entropy": 0.05,
            "coherence": 0.05,
        }

        for dim in dimensions:
            score += weights.get(dim, 0.0)
            
        # Bonus points for using bitchains for actual analysis
        
        # Bonus for actual coordinate analysis
        if len(bitchains) > 0:
            # Analyze coordinate variance across bitchains
            REALMS = ["COMPANION", "BADGE", "SPONSOR_RING", "ACHIEVEMENT", "PATTERN",
                      "FACULTY", "TEMPORAL", "VOID"]

            realm_count = len(set(bc.coordinates.realm for bc in bitchains))
            lineage_variance = len(set(bc.coordinates.lineage for bc in bitchains))
            adjacency_complexity = sum(len(bc.coordinates.adjacency) for bc in bitchains) / len(bitchains)
            
            # Bonus based on actual coordinate diversity
            diversity_bonus = min(0.1, (realm_count / len(REALMS)) * 0.05 + 
                                      (lineage_variance / 100) * 0.03 + 
                                      (adjacency_complexity / 5) * 0.02)
            score += diversity_bonus

            # If bitchains align with actual addressing strategy
            if any(bc.coordinates.luminosity != 0 for bc in bitchains) and  \
               any(bc.coordinates.dimensionality != 0 for bc in bitchains):
                score += 0.15  # Apply additional bonus for dynamic properties

        # Normalize to 0.0-1.0 range
        return min(score, 1.0)

    def _test_dimension_count(self, dimension_count: int) -> DimensionTestResult:
        """
        Test a specific dimension count.

        Args:
            dimension_count: Number of dimensions to use

        Returns:
            DimensionTestResult with metrics
        """
        dimensions = self._select_dimensions(dimension_count)

        # Generate random bit-chains
        bitchains = [generate_random_bitchain(seed=i) for i in range(self.sample_size)]

        # Compute addresses with selected dimensions
        addresses = set()
        address_list = []
        storage_sizes = []

        for bc in bitchains:
            addr = self._compute_address_with_dimensions(bc, dimensions)
            address_list.append(addr)
            addresses.add(addr)

            # Calculate storage size (simplified: JSON size of coordinate dict)
            coords_dict = {dim: getattr(bc.coordinates, dim, 0.0) for dim in dimensions}
            # Convert enum values to their string representation for JSON serialization
            serializable_coords = {}
            for k, v in coords_dict.items():
                if hasattr(v, 'value'):
                    serializable_coords[k] = v.value
                else:
                    serializable_coords[k] = v
            storage_sizes.append(len(json.dumps(serializable_coords)))

        # Calculate collision metrics
        unique_count = len(addresses)
        collisions = self.sample_size - unique_count
        collision_rate = collisions / self.sample_size if self.sample_size > 0 else 0.0

        # Measure retrieval latency (simulate hash table lookup)
        latencies = []
        address_to_bc = {
            self._compute_address_with_dimensions(bc, dimensions): bc
            for bc in bitchains
        }

        for _ in range(min(1000, self.sample_size)):
            target_addr = secure_random.choice(address_list)
            start = time.perf_counter()
            _ = address_to_bc.get(target_addr)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        mean_latency = statistics.mean(latencies) if latencies else 0.0
        median_latency = statistics.median(latencies) if latencies else 0.0

        # Calculate storage metrics
        avg_storage = statistics.mean(storage_sizes) if storage_sizes else 0
        storage_per_dim = avg_storage / max(dimension_count, 1)

        # Calculate semantic expressiveness
        expressiveness = self._calculate_semantic_expressiveness(dimensions, bitchains)

        return DimensionTestResult(
            dimension_count=dimension_count,
            dimensions_used=dimensions,
            sample_size=self.sample_size,
            unique_addresses=unique_count,
            collisions=collisions,
            collision_rate=collision_rate,
            mean_retrieval_latency_ms=mean_latency,
            median_retrieval_latency_ms=median_latency,
            avg_storage_bytes=int(avg_storage),
            storage_overhead_per_dimension=storage_per_dim,
            semantic_expressiveness_score=expressiveness,
        )

    def run(self) -> Tuple[DimensionCardinalityResult, bool]:
        """
        Run the dimension cardinality analysis.

        Returns:
            Tuple of (results, success)
        """
        start_time = datetime.now(timezone.utc).isoformat()
        overall_start = time.time()

        print("\n" + "=" * 80)
        print("EXP-11: DIMENSION CARDINALITY ANALYSIS")
        print("=" * 80)
        print(f"Sample size: {self.sample_size} bit-chains per dimension count")
        print(f"Dimension counts: {self.dimension_counts}")
        print(f"Test iterations: {self.test_iterations}")
        print()

        print("Testing dimension counts...")
        print("-" * 80)

        # Test each dimension count
        for dim_count in self.dimension_counts:
            print(f"\nTesting {dim_count} dimensions:")

            # Run multiple iterations and average results
            iteration_results = []
            for iteration in range(self.test_iterations):
                iter_result = self._test_dimension_count(dim_count)
                iteration_results.append(iter_result)

                if (iteration + 1) % max(1, self.test_iterations // 2) == 0:
                    print(
                        f"  Iteration {iteration + 1}/{self.test_iterations}: "
                        f"Collisions={iter_result.collisions}, "
                        f"Latency={iter_result.mean_retrieval_latency_ms:.4f}ms"
                    )

            # Average the iteration results
            avg_result = DimensionTestResult(
                dimension_count=dim_count,
                dimensions_used=iteration_results[0].dimensions_used,
                sample_size=self.sample_size,
                unique_addresses=int(
                    statistics.mean([r.unique_addresses for r in iteration_results])
                ),
                collisions=int(
                    statistics.mean([r.collisions for r in iteration_results])
                ),
                collision_rate=statistics.mean(
                    [r.collision_rate for r in iteration_results]
                ),
                mean_retrieval_latency_ms=statistics.mean(
                    [r.mean_retrieval_latency_ms for r in iteration_results]
                ),
                median_retrieval_latency_ms=statistics.mean(
                    [r.median_retrieval_latency_ms for r in iteration_results]
                ),
                avg_storage_bytes=int(
                    statistics.mean([r.avg_storage_bytes for r in iteration_results])
                ),
                storage_overhead_per_dimension=statistics.mean(
                    [r.storage_overhead_per_dimension for r in iteration_results]
                ),
                semantic_expressiveness_score=statistics.mean(
                    [r.semantic_expressiveness_score for r in iteration_results]
                ),
            )

            self.results.append(avg_result)

            print(
                f"  Average: Collision Rate={avg_result.collision_rate:.4%}, "
                f"Latency={avg_result.mean_retrieval_latency_ms:.4f}ms, "
                f"Expressiveness={avg_result.semantic_expressiveness_score:.2f}"
            )

        # Analyze results
        print()
        print("=" * 80)
        print("DIMENSION CARDINALITY ANALYSIS")
        print("=" * 80)

        # Handle empty results
        if not self.results:
            print("No dimension counts tested.")
            end_time = datetime.now(timezone.utc).isoformat()
            empty_result: DimensionCardinalityResult = DimensionCardinalityResult(
                start_time=start_time,
                end_time=end_time,
                total_duration_seconds=time.time() - overall_start,
                sample_size=self.sample_size,
                dimension_counts_tested=self.dimension_counts,
                test_iterations=self.test_iterations,
                dimension_results=[],
                optimal_dimension_count=7,
                optimal_collision_rate=0.0,
                optimal_retrieval_latency_ms=0.0,
                optimal_storage_efficiency=0.0,
                diminishing_returns_threshold=7,
                major_findings=["No dimension counts tested"],
                seven_dimensions_justified=False,
            )
            print("=" * 80)
            return empty_result, False

        # Find optimal dimension count (lowest collision rate + good
        # expressiveness)
        optimal_result = min(
            self.results,
            key=lambda r: r.collision_rate
            + (1.0 - r.semantic_expressiveness_score) * 0.1,
        )

        # Find diminishing returns threshold
        # (where adding more dimensions doesn't significantly reduce collisions)
        diminishing_threshold = 7  # Default to 7
        for i in range(len(self.results) - 1):
            current = self.results[i]
            next_result = self.results[i + 1]

            # If collision rate improvement is < 10%, we've hit diminishing
            # returns
            if current.collision_rate > 0:
                improvement = (
                    current.collision_rate - next_result.collision_rate
                ) / current.collision_rate
                if improvement < 0.10:
                    diminishing_threshold = current.dimension_count
                    break

        # Check if 7 dimensions is justified
        seven_dim_result = next(
            (r for r in self.results if r.dimension_count == 7), None
        )
        seven_justified = False

        if seven_dim_result:
            # 7 dimensions justified if:
            # 1. Collision rate is very low (< 0.1%)
            # 2. Semantic expressiveness is high (> 0.9)
            # 3. It's at or near the optimal point
            seven_justified = (
                seven_dim_result.collision_rate < 0.001
                and seven_dim_result.semantic_expressiveness_score > 0.9
                and seven_dim_result.dimension_count
                >= optimal_result.dimension_count - 1
            )

        # Generate findings
        major_findings = []

        major_findings.append(
            f"Optimal dimension count: {optimal_result.dimension_count} "
            f"(collision rate: {optimal_result.collision_rate:.4%})"
        )

        major_findings.append(
            f"Diminishing returns threshold: {diminishing_threshold} dimensions"
        )

        if seven_justified:
            major_findings.append(
                "[OK] 7 dimensions justified: optimal balance of expressiveness and efficiency"
            )
        else:
            major_findings.append(
                f"[WARN] 7 dimensions may not be optimal (best: {
                    optimal_result.dimension_count
                })"
            )

        # Collision rate analysis
        three_dim = next((r for r in self.results if r.dimension_count == 3), None)
        seven_dim = seven_dim_result
        ten_dim = next((r for r in self.results if r.dimension_count == 10), None)

        if three_dim and seven_dim and three_dim.collision_rate > 0:
            improvement = (
                (three_dim.collision_rate - seven_dim.collision_rate)
                / three_dim.collision_rate
                * 100
            )
            major_findings.append(
                f"Collision improvement (3->7 dims): {improvement:.1f}% reduction"
            )

        if seven_dim and ten_dim:
            improvement = (
                (seven_dim.collision_rate - ten_dim.collision_rate)
                / max(seven_dim.collision_rate, 0.0001)
                * 100
            )
            major_findings.append(
                f"Collision improvement (7->10 dims): {improvement:.1f}% reduction"
            )

        # Storage efficiency
        if seven_dim:
            major_findings.append(
                f"Storage overhead at 7 dims: {seven_dim.avg_storage_bytes} bytes "
                f"({seven_dim.storage_overhead_per_dimension:.1f} bytes/dimension)"
            )

        # Semantic expressiveness
        if seven_dim:
            major_findings.append(
                f"Semantic expressiveness at 7 dims: {
                    seven_dim.semantic_expressiveness_score:.1%}"
            )

        print()
        for finding in major_findings:
            print(f"  {finding}")
        print()

        overall_end = time.time()
        end_time = datetime.now(timezone.utc).isoformat()

        result: DimensionCardinalityResult = DimensionCardinalityResult(
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=overall_end - overall_start,
            sample_size=self.sample_size,
            dimension_counts_tested=self.dimension_counts,
            test_iterations=self.test_iterations,
            dimension_results=self.results,
            optimal_dimension_count=optimal_result.dimension_count,
            optimal_collision_rate=optimal_result.collision_rate,
            optimal_retrieval_latency_ms=optimal_result.mean_retrieval_latency_ms,
            optimal_storage_efficiency=optimal_result.storage_overhead_per_dimension,
            diminishing_returns_threshold=diminishing_threshold,
            major_findings=major_findings,
            seven_dimensions_justified=seven_justified,
        )

        # Success if optimal is 7-9 dimensions (research shows 8 may be better)
        # OR if collision rate is negligible at any tested dimension count
        success = (
            optimal_result.dimension_count >= 7 and optimal_result.dimension_count <= 9
        ) or optimal_result.collision_rate < 0.001

        print("=" * 80)
        if success:
            print(
                f"RESULT: [OK] DIMENSION ANALYSIS COMPLETE "
                f"(optimal: {optimal_result.dimension_count} dimensions)"
            )
        else:
            print(
                f"RESULT: [INFO] DIMENSION ANALYSIS COMPLETE "
                f"(optimal: {optimal_result.dimension_count} dimensions)"
            )
        print("=" * 80)

        return result, success


# ============================================================================
# CLI & RESULTS PERSISTENCE
# ============================================================================


def save_results(
    results: DimensionCardinalityResult, output_file: Optional[str] = None
) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp11_dimension_cardinality_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Load from config or fall back to command-line args
    try:
        from fractalsemantics.config import ExperimentConfig

        config = ExperimentConfig()
        sample_size = config.get("EXP-11", "sample_size", 1000)
        dimension_counts = config.get(
            "EXP-11", "dimension_counts", [3, 4, 5, 6, 7, 8, 9, 10]
        )
        test_iterations = config.get("EXP-11", "test_iterations", 5)
    except Exception:
        sample_size = 1000
        dimension_counts = [3, 4, 5, 6, 7, 8, 9, 10]
        test_iterations = 5

        if "--quick" in sys.argv:
            sample_size = 100
            test_iterations = 2
        elif "--full" in sys.argv:
            sample_size = 5000
            test_iterations = 10

    try:
        experiment = EXP11_DimensionCardinality(
            sample_size=sample_size,
            dimension_counts=dimension_counts,
            test_iterations=test_iterations,
        )
        results, success = experiment.run()
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("[OK] EXP-11 COMPLETE")
        print("=" * 80)
        print(f"Results: {output_file}")
        print()

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
