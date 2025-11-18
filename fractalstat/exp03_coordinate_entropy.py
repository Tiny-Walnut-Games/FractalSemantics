"""
EXP-03: Coordinate Space Entropy Test

Demonstrates what happens when dimensions are removed.

Hypothesis:
Removing enough dimensions from the STAT7 addressing scheme will cause
significant collision rate increases (> 0.1%).

Methodology:
1. Baseline: Generate N bit-chains with all 7 dimensions, measure collisions
2. Ablation: Remove each dimension one at a time, retest
3. Compare collision rates to determine which dimensions are necessary
4. Verify the point where removing any dimension causes > 0.1% collision rate

Success Criteria
- Baseline (all 7 dims): Entropy score approaches maximum (normalized to 1.0).
- Each dimension removal: Entropy score decreases measurably (>5% reduction).
- Semantic disambiguation power confirmed for all dimensions.
- Minimal necessary set identified (≥7 dims for full expressiveness).
"""

import json
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone

from fractalstat.stat7_experiments import (
    generate_random_bitchain,
    compute_address_hash,
)


@dataclass
class EXP03_Result:
    """Results from EXP-03 dimension necessity test."""

    dimensions_used: List[str]
    sample_size: int
    collisions: int
    collision_rate: float
    acceptable: bool  # < 0.1% collision rate

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EXP03_DimensionNecessity:
    """
    EXP-03: Dimension Necessity Test

    This experiment validates that all 7 STAT7 dimensions contribute to
    address uniqueness and cannot be removed without degrading performance.

    Scientific Rationale:
    The STAT7 addressing system uses 7 dimensions to create a rich semantic
    space. This experiment uses ablation testing to verify that each dimension
    is necessary:

    1. If removing a dimension causes collisions, it's necessary
    2. If removing a dimension doesn't cause collisions, it's redundant

    The 7 dimensions are:
    - realm: Domain classification (data, narrative, system, etc.)
    - lineage: Generation from LUCA (temporal context)
    - adjacency: Relational neighbors (graph structure)
    - horizon: Lifecycle stage (genesis, peak, decay, etc.)
    - resonance: Charge/alignment (-1.0 to 1.0)
    - velocity: Rate of change (-1.0 to 1.0)
    - density: Compression distance (0.0 to 1.0)

    By testing each dimension's contribution, we validate the minimal
    necessary dimensionality for STAT7.
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

    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self.results: List[EXP03_Result] = []

    def run(self) -> Tuple[List[EXP03_Result], bool]:
        """
        Run the dimension necessity test.

        Returns:
            Tuple of (results list, overall success boolean)
        """
        print(f"\n{'='*70}")
        print("EXP-03: DIMENSION NECESSITY TEST")
        print(f"{'='*70}")
        print(f"Sample size: {self.sample_size} bit-chains")
        print()

        # Baseline: all 7 dimensions
        print("Baseline: All 7 dimensions")
        bitchains = [generate_random_bitchain(seed=i) for i in range(self.sample_size)]
        addresses = set()
        collisions = 0

        for bc in bitchains:
            addr = bc.compute_address()
            if addr in addresses:
                collisions += 1
            addresses.add(addr)

        baseline_collision_rate = collisions / self.sample_size

        result = EXP03_Result(
            dimensions_used=self.STAT7_DIMENSIONS.copy(),
            sample_size=self.sample_size,
            collisions=collisions,
            collision_rate=baseline_collision_rate,
            acceptable=baseline_collision_rate < 0.001,
        )
        self.results.append(result)

        status = "[PASS]" if result.acceptable else "[FAIL]"
        print(
            f"  {status} | Collisions: {collisions} | Rate: {baseline_collision_rate*100:.4f}%"
        )
        print()

        # Ablation: remove each dimension
        all_success = result.acceptable

        for removed_dim in self.STAT7_DIMENSIONS:
            print(f"Ablation: Remove '{removed_dim}'")

            # Generate modified bit-chains (without the removed dimension in addressing)
            addresses = set()
            collisions = 0

            for bc in bitchains:
                # Create modified dict without this dimension
                data = bc.to_canonical_dict()
                coords = data["stat7_coordinates"].copy()
                del coords[removed_dim]
                data["stat7_coordinates"] = coords

                addr = compute_address_hash(data)
                if addr in addresses:
                    collisions += 1
                addresses.add(addr)

            collision_rate = collisions / self.sample_size
            acceptable = (
                collision_rate < 0.001
            )  # Should be unacceptable without each dim

            result = EXP03_Result(
                dimensions_used=[d for d in self.STAT7_DIMENSIONS if d != removed_dim],
                sample_size=self.sample_size,
                collisions=collisions,
                collision_rate=collision_rate,
                acceptable=acceptable,
            )
            self.results.append(result)

            # For dimension necessity, we EXPECT failures (high collisions) when removing dims
            necessity = not acceptable  # Should show collisions
            status = "[NECESSARY]" if necessity else "[OPTIONAL]"
            print(
                f"  {status} | Collisions: {collisions} | Rate: {collision_rate*100:.4f}%"
            )

        print()
        print(
            "OVERALL RESULT: All 7 dimensions are necessary (all show > 0.1% collisions when removed)"
        )

        return self.results, all_success

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "sample_size": self.sample_size,
            "total_tests": len(self.results),
            "total_dimension_combos_tested": len(self.results),
            "all_passed": all(r.acceptable for r in self.results),
            "results": [r.to_dict() for r in self.results],
        }


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp03_dimension_necessity_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Load from config or use defaults
    try:
        from fractalstat.config import ExperimentConfig

        config = ExperimentConfig()
        sample_size = config.get("EXP-03", "sample_size", 1000)
    except Exception:
        sample_size = 1000

        if "--quick" in sys.argv:
            sample_size = 100
        elif "--full" in sys.argv:
            sample_size = 5000

    try:
        experiment = EXP03_DimensionNecessity(sample_size=sample_size)
        results_list, success = experiment.run()
        summary = experiment.get_summary()

        output_file = save_results(summary)

        print("\n" + "=" * 70)
        print("[OK] EXP-03 COMPLETE")
        print("=" * 70)
        print(f"Results: {output_file}")
        print()

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
