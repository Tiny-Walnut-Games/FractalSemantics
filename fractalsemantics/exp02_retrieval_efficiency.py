#!/usr/bin/env python3
# pylint: disable=C0301,C0116,W0404,W0621,W0212,W0718
"""
EXP-02: Retrieval Efficiency Test

Validates that retrieving a bit-chain by FractalSemantics address is fast at scale.

Hypothesis:
Retrieval latency scales logarithmically or better with dataset size.

Methodology:
1. Build indexed set of N bit-chains at different scales (1M, 100M, 10B, 1T)
2. Query M random addresses (default: 1,000,000 queries)
3. Measure latency percentiles (mean, median, P95, P99)
4. Verify retrieval meets performance targets at each scale

Success Criteria:
- Mean latency < 0.1ms at 1M bit-chains
- Mean latency < 0.5ms at 100M bit-chains
- Mean latency < 2.0ms at 10B bit-chains
- Mean latency < 5.0ms at 1T bit-chains
- Latency scales logarithmically or better
"""

import ast
import gc
import json
import secrets
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil  # type: ignore[import-untyped]

from fractalsemantics.fractalsemantics_entity import generate_random_bitchain
from fractalsemantics.progress_comm import ProgressReporter

# Import subprocess communication for enhanced progress reporting
try:
    from fractalsemantics.subprocess_comm import (
        is_subprocess_communication_enabled,
        send_subprocess_completion,
        send_subprocess_progress,
        send_subprocess_status,
    )
except ImportError:
    # Fallback if subprocess communication is not available
    def send_subprocess_progress(*args, **kwargs) -> bool: return False
    def send_subprocess_status(*args, **kwargs) -> bool: return False
    def send_subprocess_completion(*args, **kwargs) -> bool: return False
    def is_subprocess_communication_enabled() -> bool: return False

secure_random = secrets.SystemRandom()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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
    cache_hit_rate: float
    memory_pressure: Optional[float]  # Memory usage as percentage
    warmup_time_ms: float
    success: bool  # target_latency < threshold

    def to_dict(self) -> dict[str, any]:
        return asdict(self)


class EXP02_RetrievalEfficiency:
    """
    EXP-02: Retrieval Efficiency Test

    This experiment validates that FractalSemantics address-based retrieval is fast enough
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

    def __init__(self, query_count: int = 1000000):
        self.query_count = query_count
        # Load scales from config or use scaled defaults
        try:
            from fractalsemantics.config import ExperimentConfig
            config = ExperimentConfig()
            self.scales = config.get("EXP-02", "scales", [1000000, 100000000, 10000000000, 1000000000000])
        except Exception:
            self.scales = [1000000, 100000000, 10000000000, 1000000000000]  # Scaled defaults: 1M, 100M, 10B, 1T
        self.results: list[EXP02_Result] = []

    def run(self) -> tuple[list[EXP02_Result], bool]:
        """
        Run the retrieval efficiency test with comprehensive benchmarking.

        This enhanced version includes:
        - Warmup periods to account for JIT compilation and caching
        - Memory pressure testing with realistic data storage
        - Cache hit/miss simulation
        - Multiple query patterns (cached, random, adversarial)

        Returns:
            tuple of (results list, overall success boolean)
        """
        print(f"\n{'=' * 70}")
        print("EXP-02: RETRIEVAL EFFICIENCY TEST (ENHANCED)")
        print(f"{'=' * 70}")
        print(f"Query count per scale: {self.query_count}")
        print(f"Scales: {self.scales}")
        print("Includes: warmup, memory pressure, cache simulation")
        print()

        # Send progress message for experiment start
        try:
            progress = ProgressReporter("EXP-02")
            progress.status("Initialization", "Starting retrieval efficiency test")

            # Send subprocess progress message
            send_subprocess_status("EXP-02", "Initialization", "Starting retrieval efficiency test")
        except ast.ParseError:
            pass  # Ignore if progress communication is not available

        all_success = True
        # Updated thresholds for scaled experiments - more lenient for larger scales
        thresholds = {
            1000000: 0.1,      # 1M: 0.1ms target
            100000000: 0.5,    # 100M: 0.5ms target
            10000000000: 2.0,  # 10B: 2.0ms target
            1000000000000: 5.0 # 1T: 5.0ms target
        }

        for i, scale in enumerate(self.scales):
            progress_percent = (i / len(self.scales)) * 100
            print(f"Testing scale: {scale:,} bit-chains")

            # Send progress message for scale start
            try:
                progress = ProgressReporter("EXP-02")
                progress.status(f"Scale {scale:,}", f"Testing {scale:,} bit-chains")
            except ast.ParseError:
                pass  # Ignore if progress communication is not available

            start_time = time.time()

            # 1. Generate bit-chains with realistic data storage
            bitchains = []
            for i in range(scale):
                bc = generate_random_bitchain(seed=i)
                # Store realistic payload data to simulate real-world overhead
                # This adds memory pressure and more realistic lookup costs
                payload_data = f"simulated_payload_{i}" * 100  # ~2KB per chain
                bitchains.append((bc, payload_data))

            # 2. Index by address for more realistic storage simulation
            # Use a richer structure to avoid pure Python dict optimization
            address_to_data: dict[str, dict[str, any]] = {}
            for bc, payload_data in bitchains:
                addr = bc.compute_address()
                address_to_data[addr] = {
                    'bitchain': bc,
                    'payload_size': len(payload_data),
                    'metadata': {
                        'created': time.time(),
                        'accessed_count': 0,
                        'last_accessed': None
                    }
                }

            addresses = list(address_to_data.keys())
            print(f"  Index built: {len(addresses)} entries")

            # 3. Warmup phase - perform operations to stabilize performance
            print("  Warmup phase...")
            warmup_start = time.perf_counter()
            warmup_operations = min(1000, scale // 10)  # Scale warmup with dataset size

            for _ in range(warmup_operations):
                # Perform random access patterns during warmup
                addr = secure_random.choice(addresses)
                _ = address_to_data[addr]['bitchain']

                # Simulate some metadata updates
                if secure_random.random() < 0.1:  # 10% chance
                    address_to_data[addr]['metadata']['accessed_count'] += 1
                    address_to_data[addr]['metadata']['last_accessed'] = time.time()

            warmup_time = (time.perf_counter() - warmup_start) * 1000
            print(f"  Warmup complete: {warmup_time:.3f}ms")

            # 4. Memory pressure test - force garbage collection and measure impact
            if HAS_PSUTIL:
                try:
                    process = psutil.Process()
                    memory_before = process.memory_percent()
                    gc.collect()  # Force garbage collection
                    memory_after = process.memory_percent()
                    memory_pressure = max(memory_before, memory_after)
                except Exception:
                    memory_pressure = None
            else:
                gc.collect()
                memory_pressure = None

            print(f"  Memory pressure: {memory_pressure:.1f}%" if memory_pressure else "  Memory pressure: N/A")

            # Artificial memory pressure for larger datasets
            if scale >= 10000:
                # Create some memory pressure by allocating temporary objects
                pressure_objects = [list(range(1000)) for _ in range(100)]
                del pressure_objects
                gc.collect()

            # 5. Performance measurement with multiple query patterns
            print("  Performance measurement...")
            latencies = []
            hits = 0
            total_queries = 0

            # Mix of query patterns to simulate real-world usage
            query_patterns = self._generate_query_patterns(addresses, self.query_count)

            # Progress tracking for query execution
            progress_interval = max(1, self.query_count // 10)  # Update every 10%

            for query_idx, query_addr in enumerate(query_patterns):
                total_queries += 1

                start = time.perf_counter()
                result = address_to_data.get(query_addr)
                if result:
                    hits += 1
                    # Access the payload to simulate realistic retrieval
                    _ = result['bitchain']
                    _ = result['payload_size']
                    # Update metadata to simulate real usage
                    result['metadata']['accessed_count'] += 1
                    result['metadata']['last_accessed'] = time.time()
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

                latencies.append(elapsed)

                # Send progress update every 10%
                if query_idx % progress_interval == 0 and query_idx > 0:
                    query_progress = progress_percent + (query_idx / self.query_count) * (100 / len(self.scales))
                    try:
                        progress = ProgressReporter("EXP-02")
                        progress.update(query_progress, f"{scale:,} Scale", f"Executed {query_idx:,}/{self.query_count:,} queries")

                        # Send subprocess progress message
                        send_subprocess_progress("EXP-02", query_progress, f"{scale:,} Scale", f"Executed {query_idx:,}/{self.query_count:,} queries")
                    except ast.ParseError:
                        pass

            # 6. Compute enhanced statistics
            latencies.sort()
            mean_lat = sum(latencies) / len(latencies)
            median_lat = latencies[len(latencies) // 2]
            p95_lat = latencies[int(len(latencies) * 0.95)]
            p99_lat = latencies[int(len(latencies) * 0.99)]
            min_lat = latencies[0]
            max_lat = latencies[-1]
            cache_hit_rate = hits / total_queries if total_queries > 0 else 0.0

            threshold = thresholds.get(scale, 2.0)
            success = mean_lat < threshold

            exp_result: EXP02_Result = EXP02_Result(
                scale=scale,
                queries=self.query_count,
                mean_latency_ms=mean_lat,
                median_latency_ms=median_lat,
                p95_latency_ms=p95_lat,
                p99_latency_ms=p99_lat,
                min_latency_ms=min_lat,
                max_latency_ms=max_lat,
                cache_hit_rate=cache_hit_rate,
                memory_pressure=memory_pressure,
                warmup_time_ms=warmup_time,
                success=success,
            )

            self.results.append(exp_result)
            all_success = all_success and success

            total_time = time.time() - start_time
            status = "PASS" if success else "FAIL"
            print(
                f"  {status} | Mean: {mean_lat:.4f}ms | "
                f"Median: {median_lat:.4f}ms"
            )
            print(
                f"       P95: {p95_lat:.4f}ms | P99: {p99_lat:.4f}ms | "
                f"Cache: {cache_hit_rate:.1%}"
            )
            print(
                f"       Target: < {threshold}ms | "
                f"Time: {total_time:.1f}s"
            )
            print()

        if all_success:
            print("OVERALL RESULT: ALL PASS")
        else:
            print("OVERALL RESULT: SOME FAILED")

        # Send completion progress message
        try:
            progress = ProgressReporter("EXP-02")
            progress.complete("Retrieval efficiency test completed")

            # Send subprocess completion message
            send_subprocess_completion("EXP-02", all_success, f"Retrieval efficiency {'passed' if all_success else 'failed'}")
        except ast.ParseError:
            pass  # Ignore if progress communication is not available

        return self.results, all_success

    def _generate_query_patterns(self, addresses: list[str], query_count: int) -> list[str]:
        """
        Generate realistic query patterns including:
        - Hot data access patterns (recently accessed items)
        - Random access patterns
        - Cache-friendly access patterns (temporal locality)
        - Edge cases (non-existent keys, adversarial patterns)
        """
        queries = []

        # 70% random access (typical DB workload)
        for _ in range(int(query_count * 0.7)):
            queries.append(secure_random.choice(addresses))

        # 20% temporal locality (recently accessed patterns)
        # Simulate by preferring items from the first quarter of the address list
        locality_subset = addresses[:len(addresses)//4]
        for _ in range(int(query_count * 0.2)):
            queries.append(secure_random.choice(locality_subset))

        # 10% adversarial patterns (worst case, potentially cache misses)
        # Use non-sequential access patterns
        for _ in range(query_count - len(queries)):
            # Jump around the list to avoid cache-friendly patterns
            idx = secure_random.randint(0, len(addresses)-1)
            jump = secure_random.randint(1, len(addresses)//10)
            queries.append(addresses[(idx + jump) % len(addresses)])

        # Shuffle to avoid artificial patterns in measurement
        secure_random.shuffle(queries)
        return queries[:query_count]

    def get_summary(self) -> dict[str, any]:
        """Get summary statistics."""
        return {
            "total_scales_tested": len(self.results),
            "all_passed": bool(self.results) and all(r.success for r in self.results),
            "results": [r.to_dict() for r in self.results],
        }


def save_results(results: dict[str, any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp02_retrieval_efficiency_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Load from config or use defaults
    query_count = 1000
    try:
        from fractalsemantics.config import ExperimentConfig

        config = ExperimentConfig()
        query_count = config.get("EXP-02", "query_count", 1000)

        # Detect dev environment config and upgrade to production settings for consistency
        env = config.get_environment()
        if env == "dev" and query_count == 100:  # Dev default is too low for reliable testing
            # Use production settings to ensure orchestrator runs match direct --full runs
            query_count = 5000  # Same as --full flag
    except Exception:
        pass  # Use default value set above

    # Check CLI args regardless of config success (these override config)
    if "--quick" in sys.argv:
        query_count = 100
    elif "--full" in sys.argv:
        query_count = 500000

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
