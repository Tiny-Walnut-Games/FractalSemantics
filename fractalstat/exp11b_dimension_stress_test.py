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
import random

# Reuse canonical serialization from Phase 1
from fractalstat.stat7_experiments import (
    compute_address_hash,
    BitChain,
    Coordinates,
    REALMS,
    HORIZONS,
    ENTITY_TYPES,
)


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

    STAT7_DIMENSIONS = [
        "realm",
        "lineage",
        "adjacency",
        "horizon",
        "resonance",
        "velocity",
        "density",
    ]

    def __init__(self, sample_size: int = 10000):
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
            realm = random.choice(REALMS[:3])  # Only first 3 realms
            lineage = random.randint(1, 10)  # Only 1-10 instead of 1-100
            adjacency = []  # No adjacency relationships
            horizon = random.choice(HORIZONS[:2])  # Only first 2 horizons
            resonance = random.uniform(
                -coordinate_range_limit, coordinate_range_limit
            )  # Limited range
            velocity = random.uniform(-coordinate_range_limit, coordinate_range_limit)
            density = random.uniform(
                0.5 - coordinate_range_limit, 0.5 + coordinate_range_limit
            )
        else:
            # Full range
            realm = random.choice(REALMS)
            lineage = random.randint(1, 100)
            adjacency_count = random.randint(0, 5)
            adjacency = [f"adj-{index}-{i}" for i in range(adjacency_count)]
            horizon = random.choice(HORIZONS)
            resonance = random.uniform(-1.0, 1.0)
            velocity = random.uniform(-1.0, 1.0)
            density = random.uniform(0.0, 1.0)

        coords = Coordinates(
            realm=realm,
            lineage=lineage,
            adjacency=adjacency,
            horizon=horizon,
            resonance=resonance,
            velocity=velocity,
            density=density,
        )

        return BitChain(
            id=id_str,
            entity_type=random.choice(ENTITY_TYPES),
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
            elif dim == "resonance":
                coords_dict[dim] = bc.coordinates.resonance
            elif dim == "velocity":
                coords_dict[dim] = bc.coordinates.velocity
            elif dim == "density":
                coords_dict[dim] = bc.coordinates.density

        # Create data dict for hashing
        data = {
            "id": bc.id,
            "entity_type": bc.entity_type,
            "realm": bc.realm,
            "stat7_coordinates": coords_dict,
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
        Run all stress tests to find the breaking point.
        """
        start_time = datetime.now(timezone.utc).isoformat()
        overall_start = time.time()

        print("\n" + "=" * 80)
        print("EXP-11b: DIMENSIONAL COLLISION STRESS TEST")
        print("=" * 80)
        print("Goal: Find out how 'dumb' the system has to be before we see collisions")
        print(f"Sample size: {self.sample_size:,} bit-chains per test")
        print()

        # Test 1: Baseline (should have zero collisions)
        result1 = self._run_stress_test(
            test_name="Test 1: Baseline (Full System)",
            description="Unique IDs, unique state, full coordinate ranges, all 7 dimensions",
            use_unique_id=True,
            use_unique_state=True,
            coordinate_range_limit=None,
            dimensions_to_use=self.STAT7_DIMENSIONS,
        )
        self.results.append(result1)

        # Test 2: Fixed ID (coordinates must provide uniqueness)
        result2 = self._run_stress_test(
            test_name="Test 2: Fixed ID",
            description="Same ID for all, unique state, full ranges, all 7 dimensions",
            use_unique_id=False,
            use_unique_state=True,
            coordinate_range_limit=None,
            dimensions_to_use=self.STAT7_DIMENSIONS,
        )
        self.results.append(result2)

        # Test 3: Fixed ID + Fixed State (only coordinates differ)
        result3 = self._run_stress_test(
            test_name="Test 3: Fixed ID + Fixed State",
            description="Same ID, same state, full ranges, all 7 dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=None,
            dimensions_to_use=self.STAT7_DIMENSIONS,
        )
        self.results.append(result3)

        # Test 4: Limited Coordinate Range (7 dimensions)
        result4 = self._run_stress_test(
            test_name="Test 4: Limited Coordinate Range",
            description="Fixed ID, fixed state, ±10% coordinate range, all 7 dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=0.1,
            dimensions_to_use=self.STAT7_DIMENSIONS,
        )
        self.results.append(result4)

        # Test 5: Only 3 Dimensions (full range)
        result5 = self._run_stress_test(
            test_name="Test 5: Only 3 Dimensions",
            description="Fixed ID, fixed state, full ranges, only 3 dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=None,
            dimensions_to_use=["realm", "lineage", "horizon"],
        )
        self.results.append(result5)

        # Test 6: Only 2 Dimensions (full range)
        result6 = self._run_stress_test(
            test_name="Test 6: Only 2 Dimensions",
            description="Fixed ID, fixed state, full ranges, only 2 dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=None,
            dimensions_to_use=["realm", "lineage"],
        )
        self.results.append(result6)

        # Test 7: Only 1 Dimension (realm only)
        result7 = self._run_stress_test(
            test_name="Test 7: Only 1 Dimension (Realm)",
            description="Fixed ID, fixed state, full range, only realm dimension",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=None,
            dimensions_to_use=["realm"],
        )
        self.results.append(result7)

        # Test 8: Extreme Stress (3 dims + limited range)
        result8 = self._run_stress_test(
            test_name="Test 8: Extreme Stress",
            description="Fixed ID, fixed state, ±10% range, only 3 dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=0.1,
            dimensions_to_use=["realm", "lineage", "horizon"],
        )
        self.results.append(result8)

        # Test 9: Continuous dimensions only (no categorical)
        result9 = self._run_stress_test(
            test_name="Test 9: Continuous Dimensions Only",
            description="Fixed ID, fixed state, full ranges, only continuous dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=None,
            dimensions_to_use=["resonance", "velocity", "density"],
        )
        self.results.append(result9)

        # Test 10: Categorical dimensions only
        result10 = self._run_stress_test(
            test_name="Test 10: Categorical Dimensions Only",
            description="Fixed ID, fixed state, full ranges, only categorical dimensions",
            use_unique_id=False,
            use_unique_state=False,
            coordinate_range_limit=None,
            dimensions_to_use=["realm", "horizon"],
        )
        self.results.append(result10)

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
                f"Highest collision rate: {max_collision_test.collision_rate:.4%} in {max_collision_test.test_name}"
            )
            key_findings.append(
                f"  Max collisions per address: {max_collision_test.max_collisions_per_address}"
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

        result = DimensionStressTestResult(
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=overall_end - overall_start,
            test_results=self.results,
            key_findings=key_findings,
        )

        print("=" * 80)
        print("[OK] EXP-11b COMPLETE")
        print("=" * 80)

        return result, True


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

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Parse command-line arguments
    sample_size = 10000

    if "--quick" in sys.argv:
        sample_size = 1000
    elif "--full" in sys.argv:
        sample_size = 100000
    elif "--extreme" in sys.argv:
        sample_size = 1000000

    try:
        experiment = DimensionStressTest(sample_size=sample_size)
        results, success = experiment.run()
        output_file = save_results(results)

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
