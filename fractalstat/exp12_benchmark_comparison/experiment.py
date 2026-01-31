"""
EXP-12: Benchmark Comparison - Experiment Logic

This module contains the core experiment logic for benchmark comparison.
It implements the BenchmarkComparisonExperiment class that compares
FractalStat against established addressing and indexing systems including
UUID/GUID, SHA-256 content addressing, vector databases, graph databases,
and traditional RDBMS systems.

Classes:
- BenchmarkComparisonExperiment: Main experiment runner for benchmark comparison
"""

import json
import hashlib
import time
import uuid
import sys
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import statistics
import secrets

# Import core components
from fractalstat.fractalstat_entity import (
    canonical_serialize,
    compute_address_hash,
    BitChain,
    generate_random_bitchain,
)
from .entities import (
    SystemBenchmarkResult,
    BenchmarkComparisonResult,
    BenchmarkSystem,
    UUIDSystem,
    SHA256System,
    VectorDBSystem,
    GraphDBSystem,
    RDBMSSystem,
    FractalStatSystem,
)

secure_random = secrets.SystemRandom()

# ============================================================================
# BENCHMARK COMPARISON EXPERIMENT
# ============================================================================


class BenchmarkComparisonExperiment:
    """
    Compares FractalStat against common addressing/indexing systems.

    Tests:
    1. Uniqueness (collision rates)
    2. Retrieval efficiency (latency)
    3. Storage overhead
    4. Semantic expressiveness
    5. Scalability
    """

    def __init__(
        self,
        sample_size: int = 100000,
        benchmark_systems: Optional[List[str]] = None,
        scales: Optional[List[int]] = None,
        num_queries: int = 1000,
    ):
        """
        Initialize benchmark comparison experiment.

        Args:
            sample_size: Number of entities to test
            benchmark_systems: List of systems to benchmark
            scales: List of scales to test
            num_queries: Number of retrieval queries per scale
        """
        if sample_size <= 0:
            raise ValueError(f"Sample size must be positive, got {sample_size}")
        if num_queries <= 0:
            raise ValueError(f"Number of queries must be positive, got {num_queries}")

        self.sample_size = sample_size
        self.benchmark_systems = benchmark_systems or [
            "uuid",
            "sha256",
            "vector_db",
            "graph_db",
            "rdbms",
            "fractalstat",
        ]
        self.scales = scales or [10000, 100000, 1000000]
        self.num_queries = num_queries
        self.results: List[SystemBenchmarkResult] = []

    def _create_system(self, system_name: str) -> BenchmarkSystem:
        """
        Create benchmark system instance.

        Args:
            system_name: Name of the system to create

        Returns:
            BenchmarkSystem instance

        Raises:
            ValueError: If system name is unknown
        """
        systems = {
            "uuid": UUIDSystem,
            "sha256": SHA256System,
            "vector_db": VectorDBSystem,
            "graph_db": GraphDBSystem,
            "rdbms": RDBMSSystem,
            "fractalstat": FractalStatSystem,
        }

        system_class = systems.get(system_name.lower())
        if system_class is None:
            raise ValueError(f"Unknown system: {system_name}")

        return system_class()

    def _benchmark_system(
        self, system: BenchmarkSystem, scale: int
    ) -> SystemBenchmarkResult:
        """
        Benchmark a single system at a given scale.

        Args:
            system: System to benchmark
            scale: Number of entities to test

        Returns:
            SystemBenchmarkResult
        """
        print(f"  Benchmarking {system.name} at scale {scale:,}...")

        # Generate entities
        entities = [generate_random_bitchain(seed=i) for i in range(scale)]

        # Generate addresses and store
        addresses = []
        address_set = set()
        storage_sizes = []

        for entity in entities:
            addr = system.generate_address(entity)
            addresses.append(addr)
            address_set.add(addr)
            system.store(addr, entity)
            storage_sizes.append(system.get_storage_size(entity))

        # Calculate collision metrics
        unique_count = len(address_set)
        collisions = scale - unique_count
        collision_rate = collisions / scale if scale > 0 else 0.0

        # Measure retrieval latency
        latencies = []

        for _ in range(min(self.num_queries, scale)):
            target_addr = secure_random.choice(addresses)
            start = time.perf_counter()
            _ = system.retrieve(target_addr)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        latencies.sort()
        mean_latency = statistics.mean(latencies) if latencies else 0.0
        median_latency = statistics.median(latencies) if latencies else 0.0
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        # Calculate storage metrics
        avg_storage = statistics.mean(storage_sizes) if storage_sizes else 0
        total_storage = sum(storage_sizes)

        # Get semantic capabilities
        semantic_expr = system.get_semantic_expressiveness()
        relationship_support = system.get_relationship_support()
        query_flex = system.get_query_flexibility()

        print(
            f"    Collisions: {collisions}, "
            f"Latency: {mean_latency:.4f}ms, "
            f"Storage: {avg_storage:.0f} bytes/entity"
        )

        return SystemBenchmarkResult(
            system_name=system.name,
            scale=scale,
            num_queries=min(self.num_queries, scale),
            unique_addresses=unique_count,
            collisions=collisions,
            collision_rate=collision_rate,
            mean_retrieval_latency_ms=mean_latency,
            median_retrieval_latency_ms=median_latency,
            p95_retrieval_latency_ms=p95_latency,
            p99_retrieval_latency_ms=p99_latency,
            avg_storage_bytes_per_entity=int(avg_storage),
            total_storage_bytes=total_storage,
            semantic_expressiveness=semantic_expr,
            relationship_support=relationship_support,
            query_flexibility=query_flex,
        )

    def run(self) -> Tuple[BenchmarkComparisonResult, bool]:
        """
        Run the benchmark comparison.

        Returns:
            Tuple of (results, success)
        """
        start_time = datetime.now(timezone.utc).isoformat()
        overall_start = time.time()

        print("\n" + "=" * 80)
        print("EXP-12: BENCHMARK COMPARISON")
        print("=" * 80)
        print(f"Systems: {', '.join(self.benchmark_systems)}")
        print(f"Scales: {self.scales}")
        print(f"Queries per scale: {self.num_queries}")
        print()

        # Use largest scale that fits in sample_size
        test_scale = min(max(self.scales), self.sample_size)
        print(f"Testing at scale: {test_scale:,}")
        print("-" * 80)

        # Benchmark each system
        for system_name in self.benchmark_systems:
            try:
                system = self._create_system(system_name)
                bench_result = self._benchmark_system(system, test_scale)
                self.results.append(bench_result)
            except Exception as e:
                print(f"  [ERROR] Failed to benchmark {system_name}: {e}")

        # Comparative analysis
        print()
        print("=" * 80)
        print("COMPARATIVE ANALYSIS")
        print("=" * 80)

        # Find best systems in each category
        best_collision = min(self.results, key=lambda r: r.collision_rate)
        best_retrieval = min(self.results, key=lambda r: r.mean_retrieval_latency_ms)
        best_storage = min(self.results, key=lambda r: r.avg_storage_bytes_per_entity)
        best_semantic = max(self.results, key=lambda r: r.semantic_expressiveness)

        # Calculate overall score (weighted average)
        def overall_score(r: SystemBenchmarkResult) -> float:
            # Normalize metrics to 0-1 range (lower is better for collision/latency/storage)
            max_collision = max(res.collision_rate for res in self.results)
            max_latency = max(res.mean_retrieval_latency_ms for res in self.results)
            max_storage = max(res.avg_storage_bytes_per_entity for res in self.results)

            collision_score = (
                1.0 - (r.collision_rate / max(max_collision, 0.0001))
                if max_collision > 0
                else 1.0
            )
            latency_score = (
                1.0 - (r.mean_retrieval_latency_ms / max(max_latency, 0.0001))
                if max_latency > 0
                else 1.0
            )
            storage_score = (
                1.0 - (r.avg_storage_bytes_per_entity / max(max_storage, 1))
                if max_storage > 0
                else 1.0
            )
            semantic_score = r.semantic_expressiveness

            # Weighted average
            return (
                collision_score * 0.25
                + latency_score * 0.25
                + storage_score * 0.20
                + semantic_score * 0.30
            )

        best_overall = max(self.results, key=overall_score)

        # Find FractalStat rankings
        fractalstat_result = next((r for r in self.results if r.system_name == "FractalStat"), None)

        if fractalstat_result:
            # Rank by collision rate (1 = best)
            sorted_by_collision = sorted(self.results, key=lambda r: r.collision_rate)
            fractalstat_rank_collision = sorted_by_collision.index(fractalstat_result) + 1

            # Rank by retrieval latency
            sorted_by_latency = sorted(
                self.results, key=lambda r: r.mean_retrieval_latency_ms
            )
            fractalstat_rank_retrieval = sorted_by_latency.index(fractalstat_result) + 1

            # Rank by storage efficiency
            sorted_by_storage = sorted(
                self.results, key=lambda r: r.avg_storage_bytes_per_entity
            )
            fractalstat_rank_storage = sorted_by_storage.index(fractalstat_result) + 1

            # Rank by semantic expressiveness
            sorted_by_semantic = sorted(
                self.results,
                key=lambda r: r.semantic_expressiveness,
                reverse=True,
            )
            fractalstat_rank_semantic = sorted_by_semantic.index(fractalstat_result) + 1

            fractalstat_score = overall_score(fractalstat_result)
        else:
            fractalstat_rank_collision = len(self.results)
            fractalstat_rank_retrieval = len(self.results)
            fractalstat_rank_storage = len(self.results)
            fractalstat_rank_semantic = len(self.results)
            fractalstat_score = 0.0

        # Generate findings
        major_findings = []

        major_findings.append(
            f"Best collision rate: {best_collision.system_name} "
            f"({best_collision.collision_rate:.6%})"
        )

        major_findings.append(
            f"Best retrieval latency: {best_retrieval.system_name} "
            f"({best_retrieval.mean_retrieval_latency_ms:.4f}ms)"
        )

        major_findings.append(
            f"Best storage efficiency: {best_storage.system_name} "
            f"({best_storage.avg_storage_bytes_per_entity} bytes/entity)"
        )

        major_findings.append(
            f"Best semantic expressiveness: {best_semantic.system_name} "
            f"({best_semantic.semantic_expressiveness:.2f})"
        )

        major_findings.append(f"Best overall: {best_overall.system_name}")

        if fractalstat_result:
            major_findings.append(
                f"FractalStat rankings: Collision #{fractalstat_rank_collision}, "
                f"Retrieval #{fractalstat_rank_retrieval}, "
                f"Storage #{fractalstat_rank_storage}, "
                f"Semantic #{fractalstat_rank_semantic}"
            )

            major_findings.append(f"FractalStat overall score: {fractalstat_score:.3f}")

            # FractalStat unique strengths
            if fractalstat_rank_semantic <= 2:
                major_findings.append(
                    "[OK] FractalStat excels at semantic expressiveness (multi-dimensional addressing)"
                )

            if fractalstat_rank_collision <= 3:
                major_findings.append(
                    "[OK] FractalStat competitive on collision rates (deterministic addressing)"
                )

            # Trade-offs
            if fractalstat_rank_storage > len(self.results) // 2:
                major_findings.append(
                    "[TRADE-OFF] FractalStat has higher storage overhead (7 dimensions)"
                )

            if fractalstat_rank_retrieval <= 3:
                major_findings.append(
                    "[OK] FractalStat competitive on retrieval latency (hash-based lookup)"
                )

        # Determine if FractalStat is competitive
        fractalstat_competitive = (
            fractalstat_result is not None
            and fractalstat_rank_semantic <= 2
            and fractalstat_rank_collision <= 3
            and fractalstat_score >= 0.7
        )

        print()
        for finding in major_findings:
            print(f"  {finding}")
        print()

        overall_end = time.time()
        end_time = datetime.now(timezone.utc).isoformat()

        result = BenchmarkComparisonResult(
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=overall_end - overall_start,
            sample_size=self.sample_size,
            scales_tested=[test_scale],
            num_queries=self.num_queries,
            systems_tested=self.benchmark_systems,
            system_results=self.results,
            best_collision_rate_system=best_collision.system_name,
            best_retrieval_latency_system=best_retrieval.system_name,
            best_storage_efficiency_system=best_storage.system_name,
            best_semantic_expressiveness_system=best_semantic.system_name,
            best_overall_system=best_overall.system_name,
            fractalstat_rank_collision=fractalstat_rank_collision,
            fractalstat_rank_retrieval=fractalstat_rank_retrieval,
            fractalstat_rank_storage=fractalstat_rank_storage,
            fractalstat_rank_semantic=fractalstat_rank_semantic,
            fractalstat_overall_score=fractalstat_score,
            major_findings=major_findings,
            fractalstat_competitive=fractalstat_competitive,
        )

        # Success if FractalStat shows good semantic expressiveness (its primary strength)
        # OR if it's competitive overall (score >= 0.6)
        # Calculate FractalStat rankings first, then determine success
        if fractalstat_result:
            # Rank by semantic expressiveness
            sorted_by_semantic = sorted(
                self.results,
                key=lambda r: r.semantic_expressiveness,
                reverse=True,
            )
            fractalstat_rank_semantic = sorted_by_semantic.index(fractalstat_result) + 1
            fractalstat_score = overall_score(fractalstat_result)

            success = fractalstat_rank_semantic <= 2 or fractalstat_score >= 0.6
        else:
            success = False

        print("=" * 80)
        if success:
            print(
                f"RESULT: [OK] FractalStat DEMONSTRATES SEMANTIC STRENGTHS "
                f"(score: {fractalstat_score:.3f})"
            )
        else:
            print(
                f"RESULT: [INFO] BENCHMARK ANALYSIS COMPLETE (score: {fractalstat_score:.3f})"
            )
        print("=" * 80)

        return result, success


# ============================================================================
# CLI & RESULTS PERSISTENCE
# ============================================================================


def save_results(
    results: BenchmarkComparisonResult, output_file: Optional[str] = None
) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp12_benchmark_comparison_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main entry point for EXP-12."""
    import sys
    
    # Load from config or use defaults
    sample_size = 100000
    benchmark_systems = [
        "uuid",
        "sha256",
        "vector_db",
        "graph_db",
        "rdbms",
        "fractalstat",
    ]
    scales = [10000, 100000, 1000000]
    num_queries = 1000

    try:
        from fractalstat.config import ExperimentConfig

        config = ExperimentConfig()
        sample_size = config.get("EXP-12", "sample_size", 100000)
        benchmark_systems = config.get(
            "EXP-12",
            "benchmark_systems",
            ["uuid", "sha256", "vector_db", "graph_db", "rdbms", "fractalstat"],
        )
        scales = config.get("EXP-12", "scales", [10000, 100000, 1000000])
        num_queries = config.get("EXP-12", "num_queries", 1000)
    except Exception:
        pass  # Use default values set above

    # Check CLI args regardless of config success (these override config)
    if "--quick" in sys.argv:
        sample_size = 1000
        scales = [1000]
        num_queries = 100
    elif "--full" in sys.argv:
        sample_size = 1000000
        scales = [10000, 100000, 1000000]
        num_queries = 5000

    try:
        experiment = BenchmarkComparisonExperiment(
            sample_size=sample_size,
            benchmark_systems=benchmark_systems,
            scales=scales,
            num_queries=num_queries,
        )
        test_results, success = experiment.run()
        output_file = save_results(test_results)

        print("\n" + "=" * 80)
        print("[OK] EXP-12 COMPLETE")
        print("=" * 80)
        print(f"Results: {output_file}")
        print()

        return success

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)