#!/usr/bin/env python3
"""
EXP-11b: Dimensional Collision Stress Test

This experiment deliberately "dumbs down" the addressing system to test
the actual collision resistance provided by the dimensional structure itself,
independent of SHA-256's cryptographic guarantees.

The goal: Find out how "dumb" the system has to be before we see collisions.

Key differences from EXP-11:
1. Removes UUID dependency (uses fixed or sequential IDs)
2. Tests with similar/identical state data
3. Focuses on coordinate-only hashing
4. Uses larger sample sizes to stress-test dimensions
5. Measures collision rates when dimensions are the primary differentiator

This will reveal whether the 7-dimensional structure actually provides
meaningful collision resistance, or if we're just relying on SHA-256
doing all the heavy lifting.

Status: Experimental - Issue #37 investigation
"""

import json
import time
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import secrets

# Import core components
from fractalsemantics.fractalsemantics_entity import (
    compute_address_hash,
    BitChain,
    Coordinates,
    REALMS,
    HORIZONS,
    ENTITY_TYPES,
    POLARITY_LIST,
    ALIGNMENT_LIST,
)
from fractalsemantics.dynamic_enum import Polarity, Alignment

secure_random = secrets.SystemRandom()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Dimension constants
FRACTALSEMANTICS_DIMENSIONS = [
    "realm", "lineage", "adjacency", "horizon", "luminosity",
    "polarity", "dimensionality", "alignment"
]

# Test configuration constants
DEFAULT_SAMPLE_SIZE = 10_000
MAX_DIVERSITY_SAMPLE_SIZE = 1_000

# Coordinate range limits for stress testing
COORDINATE_RANGE_LIMITS = {
    "limited": 0.1,  # ±10%
    "very_limited": 0.01,  # ±1%
}

# Common dimension subsets for testing
DIMENSION_SUBSETS = {
    "minimal_3d": ["realm", "lineage", "horizon"],
    "minimal_2d": ["realm", "lineage"],
    "single_dimension": ["realm"],
    "continuous_only": ["luminosity", "dimensionality", "adjacency"],
    "categorical_only": ["realm", "horizon"],
    "all_dimensions": FRACTALSEMANTICS_DIMENSIONS
}

# Test scenario configurations
@dataclass
class TestScenario:
    """Configuration for a single test scenario."""
    name: str
    description: str
    use_unique_id: bool
    use_unique_state: bool
    coordinate_range_limit: Optional[float]
    dimensions: List[str]

TEST_SCENARIOS = [
    TestScenario(
        name="Test 1: Baseline (Full System)",
        description="Unique IDs, unique state, full coordinate ranges, all 8 dimensions",
        use_unique_id=True,
        use_unique_state=True,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["all_dimensions"]
    ),
    TestScenario(
        name="Test 2: Fixed ID",
        description="Same ID for all, unique state, full ranges, all 8 dimensions",
        use_unique_id=False,
        use_unique_state=True,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["all_dimensions"]
    ),
    TestScenario(
        name="Test 3: Fixed ID + Fixed State",
        description="Same ID, same state, full ranges, all 8 dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["all_dimensions"]
    ),
    TestScenario(
        name="Test 4: Limited Coordinate Range",
        description="Fixed ID, fixed state, ±10% coordinate range, all 8 dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=COORDINATE_RANGE_LIMITS["limited"],
        dimensions=DIMENSION_SUBSETS["all_dimensions"]
    ),
    TestScenario(
        name="Test 5: Only 3 Dimensions",
        description="Fixed ID, fixed state, full ranges, only 3 dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["minimal_3d"]
    ),
    TestScenario(
        name="Test 6: Only 2 Dimensions",
        description="Fixed ID, fixed state, full ranges, only 2 dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["minimal_2d"]
    ),
    TestScenario(
        name="Test 7: Only 1 Dimension (Realm)",
        description="Fixed ID, fixed state, full range, only realm dimension",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["single_dimension"]
    ),
    TestScenario(
        name="Test 8: Extreme Stress",
        description="Fixed ID, fixed state, ±10% range, only 3 dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=COORDINATE_RANGE_LIMITS["limited"],
        dimensions=DIMENSION_SUBSETS["minimal_3d"]
    ),
    TestScenario(
        name="Test 9: Continuous Dimensions Only",
        description="Fixed ID, fixed state, full ranges, only continuous dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["continuous_only"]
    ),
    TestScenario(
        name="Test 10: Categorical Dimensions Only",
        description="Fixed ID, fixed state, full ranges, only categorical dimensions",
        use_unique_id=False,
        use_unique_state=False,
        coordinate_range_limit=None,
        dimensions=DIMENSION_SUBSETS["categorical_only"]
    ),
]

# ============================================================================
# STRESS TEST DATA STRUCTURES
# ============================================================================


@dataclass
class StressTestResult:
    """Results for a single stress test configuration."""

    test_name: str
    dimension_count: int
    dimensions_used: List[str]
    sample_size: int
    unique_addresses: int
    collisions: int
    collision_rate: float
    max_collisions_per_address: int
    coordinate_diversity: float  # 0.0 to 1.0, how varied the coordinates are
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionStressTestResult:
    """Complete results from EXP-11b dimension stress testing."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    test_results: List[StressTestResult]
    key_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": "EXP-11b",
            "test_type": "Dimensional Collision Stress Test",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "key_findings": self.key_findings,
            "test_results": [r.to_dict() for r in self.test_results],
        }


# ============================================================================
# STRESS TEST EXPERIMENT
# ============================================================================


class DimensionStressTest:
    """
    Tests dimensional collision resistance by progressively "dumbing down" the system.

    Test Scenarios:
    1. Baseline: Full system with UUIDs (should have zero collisions)
    2. Fixed ID: Same ID for all bit-chains, only coordinates differ
    3. Fixed ID + Fixed State: Only coordinates provide uniqueness
    4. Limited Coordinate Range: Constrain coordinates to small ranges
    5. Minimal Dimensions: Test with 1, 2, 3 dimensions only
    6. Extreme Stress: Fixed ID, fixed state, limited ranges, few dimensions
    """

    def __init__(self, sample_size: int = DEFAULT_SAMPLE_SIZE):
        """
        Initialize the stress test runner.

        Args:
            sample_size: Number of bit-chains to generate per test scenario
        """
        if sample_size <= 0:
            raise ValueError(f"Sample size must be positive, got {sample_size}")
        self.sample_size = sample_size
        self.results: List[StressTestResult] = []

    def _generate_bitchain_with_constraints(
        self,
        index: int,
        use_unique_id: bool = True,
        use_unique_state: bool = True,
        coordinate_range_limit: Optional[float] = None,
        dimensions_to_use: Optional[List[str]] = None,
    ) -> BitChain:
        """
        Generate a bit-chain with specific constraints for stress testing.

        Args:
            index: Sequential index for deterministic generation
            use_unique_id: If False, all bit-chains get same ID
            use_unique_state: If False, all bit-chains get same state
            coordinate_range_limit: If set, limit coordinate ranges (e.g., 0.1 = ±10%)
            dimensions_to_use: If set, only vary these dimensions
        """
        # ID: unique or fixed
        if use_unique_id:
            id_str = f"test-{index:08d}"
        else:
            id_str = "fixed-id-000000"

        # State: unique or fixed
        if use_unique_state:
            state = {"value": index, "index": index}
        else:
            state = {"value": 0, "index": 0}

        # Coordinates: constrained or full range
        if coordinate_range_limit is not None:
            # Limited range around center values
            realm = secure_random.choice(REALMS[:3])  # Only first 3 realms
            lineage = secure_random.randint(1, 10)  # Only 1-10 instead of 1-100
            adjacency = []  # No adjacency relationships
            horizon = secure_random.choice(HORIZONS[:2])  # Only first 2 horizons
            luminosity = secure_random.uniform(
                -coordinate_range_limit, coordinate_range_limit
            )  # Limited range
            polarity = Polarity(secure_random.choice(POLARITY_LIST[-2:]))  # Limited polarity values
            dimensionality = int(secure_random.uniform(
                0.5 - coordinate_range_limit, 0.5 + coordinate_range_limit
            ))
            alignment = Alignment(secure_random.choice(ALIGNMENT_LIST[:2]))  # Limited alignment values
        else:
            # Full range
            realm = secure_random.choice(REALMS)
            lineage = secure_random.randint(1, 100)
            adjacency_count = secure_random.randint(0, 5)
            adjacency = [f"adj-{index}-{i}" for i in range(adjacency_count)]
            horizon = secure_random.choice(HORIZONS)
            luminosity = secure_random.uniform(0.0, 1.0)
            polarity = Polarity(secure_random.choice(POLARITY_LIST))
            dimensionality = secure_random.randint(0, 5)
            alignment = Alignment(secure_random.choice(ALIGNMENT_LIST))

        coords = Coordinates(
            realm=realm,
            lineage=lineage,
            adjacency=adjacency,
            horizon=horizon,
            luminosity=luminosity,
            polarity=polarity,
            dimensionality=dimensionality,
            alignment=alignment
        )

        return BitChain(
            id=id_str,
            entity_type=secure_random.choice(ENTITY_TYPES),
            realm=realm,
            coordinates=coords,
            created_at="2024-01-01T00:00:00.000Z",  # Fixed timestamp
            state=state,
        )

    def _compute_address_with_selected_dimensions(
        self, bc: BitChain, dimensions: List[str]
    ) -> str:
        """
        Compute address using only selected dimensions.
        This is the key function that lets us test dimensional collision resistance.
        """
        # Build coordinate dict with only selected dimensions
        coords_dict: Dict[str, Any] = {}
        for dim in dimensions:
            if dim == "realm":
                coords_dict[dim] = bc.coordinates.realm
            elif dim == "lineage":
                coords_dict[dim] = bc.coordinates.lineage
            elif dim == "adjacency":
                coords_dict[dim] = sorted(bc.coordinates.adjacency)
            elif dim == "horizon":
                coords_dict[dim] = bc.coordinates.horizon
            elif dim == "luminosity":
                coords_dict[dim] = bc.coordinates.luminosity
            elif dim == "polarity":
                coords_dict[dim] = bc.coordinates.polarity
            elif dim == "dimensionality":
                coords_dict[dim] = bc.coordinates.dimensionality
            elif dim == "alignment":
                coords_dict[dim] = bc.coordinates.alignment

        # Create data dict for hashing
        data = {
            "id": bc.id,
            "entity_type": bc.entity_type,
            "realm": bc.realm,
            "fractalsemantics_coordinates": coords_dict,
            "state": bc.state,
        }

        return compute_address_hash(data)

    def _compute_coordinate_diversity(self, bitchains: List[BitChain]) -> float:
        """
        Calculate how diverse the coordinates are (0.0 = all identical, 1.0 = maximally diverse).
        """
        if len(bitchains) < 2:
            return 1.0

        # Sample a subset for efficiency
        sample = bitchains[: min(1000, len(bitchains))]

        # Count unique values per dimension
        unique_realms = len(set(bc.coordinates.realm for bc in sample))
        unique_lineages = len(set(bc.coordinates.lineage for bc in sample))
        unique_horizons = len(set(bc.coordinates.horizon for bc in sample))

        # Normalize to 0-1 range
        realm_diversity = unique_realms / len(REALMS)
        lineage_diversity = min(unique_lineages / 100, 1.0)  # Max 100 lineages
        horizon_diversity = unique_horizons / len(HORIZONS)

        # Average diversity across dimensions
        return (realm_diversity + lineage_diversity + horizon_diversity) / 3.0

    def _run_stress_test(
        self,
        test_name: str,
        description: str,
        use_unique_id: bool,
        use_unique_state: bool,
        coordinate_range_limit: Optional[float],
        dimensions_to_use: List[str],
    ) -> StressTestResult:
        """
        Run a single stress test configuration.
        """
        print(f"\n{test_name}")
        print(f"  {description}")
        print(f"  Dimensions: {len(dimensions_to_use)} - {dimensions_to_use}")
        print(f"  Sample size: {self.sample_size:,}")

        # Generate bit-chains with constraints
        bitchains = [
            self._generate_bitchain_with_constraints(
                index=i,
                use_unique_id=use_unique_id,
                use_unique_state=use_unique_state,
                coordinate_range_limit=coordinate_range_limit,
                dimensions_to_use=dimensions_to_use,
            )
            for i in range(self.sample_size)
        ]

        # Compute addresses
        addresses: Dict[str, List[str]] = {}
        for bc in bitchains:
            addr = self._compute_address_with_selected_dimensions(bc, dimensions_to_use)
            if addr not in addresses:
                addresses[addr] = []
            addresses[addr].append(bc.id)

        # Calculate collision metrics
        unique_count = len(addresses)
        collisions = self.sample_size - unique_count
        collision_rate = collisions / self.sample_size if self.sample_size > 0 else 0.0
        max_collisions = (
            max(len(ids) for ids in addresses.values()) - 1 if addresses else 0
        )

        # Calculate coordinate diversity
        diversity = self._compute_coordinate_diversity(bitchains)

        result = StressTestResult(
            test_name=test_name,
            dimension_count=len(dimensions_to_use),
            dimensions_used=dimensions_to_use,
            sample_size=self.sample_size,
            unique_addresses=unique_count,
            collisions=collisions,
            collision_rate=collision_rate,
            max_collisions_per_address=max_collisions,
            coordinate_diversity=diversity,
            description=description,
        )

        print("  Results:")
        print(f"    Unique addresses: {unique_count:,}")
        print(f"    Collisions: {collisions:,}")
        print(f"    Collision rate: {collision_rate:.4%}")
        print(f"    Max collisions per address: {max_collisions}")
        print(f"    Coordinate diversity: {diversity:.2%}")

        return result

    def run(self) -> Tuple[DimensionStressTestResult, bool]:
        """
        Run all stress tests defined in TEST_SCENARIOS.

        Returns:
            Tuple of (results, success)
        """
        start_time = datetime.now(timezone.utc).isoformat()
        overall_start = time.time()

        print("\n" + "=" * 80)
        print("EXP-11b: DIMENSIONAL COLLISION STRESS TEST")
        print("=" * 80)
        print("Goal: Find out how 'dumb' the system has to be before we see collisions")
        print(f"Sample size: {self.sample_size:,} bit-chains per test")
        print()

        # Run all test scenarios
        for scenario in TEST_SCENARIOS:
            result = self._run_stress_test(
                test_name=scenario.name,
                description=scenario.description,
                use_unique_id=scenario.use_unique_id,
                use_unique_state=scenario.use_unique_state,
                coordinate_range_limit=scenario.coordinate_range_limit,
                dimensions_to_use=scenario.dimensions,
            )
            self.results.append(result)

        # Analyze results
        print()
        print("=" * 80)
        print("STRESS TEST ANALYSIS")
        print("=" * 80)

        key_findings = []

        # Find first test with collisions
        first_collision_test = next(
            (r for r in self.results if r.collision_rate > 0), None
        )

        if first_collision_test:
            key_findings.append(
                f"First collisions appeared in: {first_collision_test.test_name}"
            )
            key_findings.append(
                f"  Collision rate: {first_collision_test.collision_rate:.4%}"
            )
            key_findings.append(f"  Configuration: {first_collision_test.description}")
        else:
            key_findings.append(
                "No collisions detected in any test - SHA-256 is doing ALL the work!"
            )

        # Find highest collision rate
        max_collision_test = max(self.results, key=lambda r: r.collision_rate)
        if max_collision_test.collision_rate > 0:
            key_findings.append(
                f"Highest collision rate: {max_collision_test.collision_rate:.4%} in {
                    max_collision_test.test_name
                }"
            )
            key_findings.append(
                f"  Max collisions per address: {
                    max_collision_test.max_collisions_per_address
                }"
            )

        # Dimension count analysis
        dim_collision_rates = {
            r.dimension_count: r.collision_rate
            for r in self.results
            if not r.test_name.startswith("Test 1")  # Exclude baseline
        }
        if dim_collision_rates:
            key_findings.append(
                f"Collision rates by dimension count: {dim_collision_rates}"
            )

        # Key insight
        if all(r.collision_rate == 0 for r in self.results):
            key_findings.append(
                "CRITICAL INSIGHT: Even with fixed IDs, fixed state, and limited ranges, "
                "SHA-256 prevents collisions. The dimensional structure provides semantic "
                "organization, not collision resistance."
            )
        else:
            key_findings.append(
                "CRITICAL INSIGHT: Collisions only appear when we severely constrain the "
                "coordinate space. The dimensions provide meaningful differentiation, but "
                "SHA-256 is the primary collision resistance mechanism."
            )

        print()
        for finding in key_findings:
            print(f"  {finding}")
        print()

        overall_end = time.time()
        end_time = datetime.now(timezone.utc).isoformat()

        final_result: DimensionStressTestResult = DimensionStressTestResult(
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=overall_end - overall_start,
            test_results=self.results,
            key_findings=key_findings,
        )

        print("=" * 80)
        print("[OK] EXP-11b COMPLETE")
        print("=" * 80)

        return final_result, True


# ============================================================================
# CLI & RESULTS PERSISTENCE
# ============================================================================


def save_results(
    results: DimensionStressTestResult, output_file: Optional[str] = None
) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp11b_dimension_stress_test_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Load from config or use defaults
    sample_size = 10000
    try:
        from fractalsemantics.config import ExperimentConfig

        config = ExperimentConfig()
        sample_size = config.get("EXP-11b", "sample_size", 10000)
    except Exception:
        pass  # Use default value set above

    # Check CLI args regardless of config success (these override config)
    if "--quick" in sys.argv:
        sample_size = 1000
    elif "--full" in sys.argv:
        sample_size = 100000
    elif "--extreme" in sys.argv:
        sample_size = 1000000

    try:
        experiment = DimensionStressTest(sample_size=sample_size)
        test_results, success = experiment.run()
        output_file = save_results(test_results)

        print("\n" + "=" * 80)
        print("[OK] EXP-11b COMPLETE")
        print("=" * 80)
        print(f"Results: {output_file}")
        print()

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
