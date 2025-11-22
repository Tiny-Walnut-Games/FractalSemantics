"""
EXP-02: Retrieval Efficiency Test

Validates that retrieving a bit-chain by STAT7 address is fast (< 1ms) at scale.

Hypothesis:
Retrieval latency scales logarithmically or better with dataset size.

Methodology:
1. Build indexed set of N bit-chains at different scales (1K, 10K, 100K)
2. Query M random addresses (default: 1,000 queries)
3. Measure latency percentiles (mean, median, P95, P99)
4. Verify retrieval meets performance targets at each scale

Success Criteria:
- Mean latency < 0.1ms at 1,000 bit-chains
- Mean latency < 0.5ms at 10,000 bit-chains
- Mean latency < 2.0ms at 100,000 bit-chains
- Latency scales logarithmically or better
"""

import json
import sys
import time
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone

from fractalstat.stat7_experiments import generate_random_bitchain


@dataclass
class EXP02_Result:
    """Results from EXP-02 retrieval efficiency test."""

    scale: int
    queries: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    success: bool  # target_latency < threshold

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EXP02_RetrievalEfficiency:
    """
    EXP-02: Retrieval Efficiency Test

    This experiment validates that STAT7 address-based retrieval is fast enough
    for production use at realistic scales.

    Scientific Rationale:
    Hash-based retrieval should provide O(1) average-case performance, but we
    need to empirically verify that:
    1. Absolute latency is acceptable (< 1ms for most queries)
    2. Performance degrades gracefully with scale
    3. Tail latencies (P95, P99) remain reasonable

    The experiment simulates content-addressable storage using Python dict
    (hash table), which provides a realistic baseline for production systems.
    """

    def __init__(self, query_count: int = 1000):
        self.query_count = query_count
        self.scales = [1_000, 10_000, 100_000]
        self.results: List[EXP02_Result] = []

    def run(self) -> Tuple[List[EXP02_Result], bool]:
        """
        Run the retrieval efficiency test.

        Returns:
            Tuple of (results list, overall success boolean)
        """
        print(f"\n{'=' * 70}")
        print("EXP-02: RETRIEVAL EFFICIENCY TEST")
        print(f"{'=' * 70}")
        print(f"Query count per scale: {self.query_count}")
        print(f"Scales: {self.scales}")
        print()

        all_success = True
        thresholds = {1_000: 0.1, 10_000: 0.5, 100_000: 2.0}  # ms

        for scale in self.scales:
            print(f"Testing scale: {scale:,} bit-chains")

            # Generate bit-chains
            bitchains = [generate_random_bitchain(seed=i) for i in range(scale)]

            # Index by address for O(1) retrieval simulation
            address_to_bc = {bc.compute_address(): bc for bc in bitchains}
            addresses = list(address_to_bc.keys())

            # Measure retrieval latency
            latencies = []

            for _ in range(self.query_count):
                target_addr = random.choice(addresses)

                start = time.perf_counter()
                _ = address_to_bc[target_addr]  # Hash table lookup
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

                latencies.append(elapsed)

            # Compute statistics
            latencies.sort()
            mean_lat = sum(latencies) / len(latencies)
            median_lat = latencies[len(latencies) // 2]
            p95_lat = latencies[int(len(latencies) * 0.95)]
            p99_lat = latencies[int(len(latencies) * 0.99)]
            min_lat = latencies[0]
            max_lat = latencies[-1]

            threshold = thresholds.get(scale, 2.0)
            success = mean_lat < threshold

            result = EXP02_Result(
                scale=scale,
                queries=self.query_count,
                mean_latency_ms=mean_lat,
                median_latency_ms=median_lat,
                p95_latency_ms=p95_lat,
                p99_latency_ms=p99_lat,
                min_latency_ms=min_lat,
                max_latency_ms=max_lat,
                success=success,
            )

            self.results.append(result)
            all_success = all_success and success

            status = "✅ PASS" if success else "❌ FAIL"
            print(
                f"  {status} | Mean: {mean_lat:.4f}ms | "
                f"Median: {median_lat:.4f}ms | "
                f"P95: {p95_lat:.4f}ms | P99: {p99_lat:.4f}ms"
            )
            print(f"       Target: < {threshold}ms")
            print()

        print(f"OVERALL RESULT: {'✅ ALL PASS' if all_success else '❌ SOME FAILED'}")

        return self.results, all_success

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_scales_tested": len(self.results),
            "all_passed": all(r.success for r in self.results),
            "results": [r.to_dict() for r in self.results],
        }


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp02_retrieval_efficiency_{timestamp}.json"

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
        query_count = config.get("EXP-02", "query_count", 1000)
    except Exception:
        query_count = 1000

        if "--quick" in sys.argv:
            query_count = 100
        elif "--full" in sys.argv:
            query_count = 5000

    try:
        experiment = EXP02_RetrievalEfficiency(query_count=query_count)
        results_list, success = experiment.run()
        summary = experiment.get_summary()

        output_file = save_results(summary)

        print("\n" + "=" * 70)
        print("[OK] EXP-02 COMPLETE")
        print("=" * 70)
        print(f"Results: {output_file}")
        print()

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
