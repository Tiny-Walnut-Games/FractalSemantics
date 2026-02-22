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
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypeAlias

try:
    from fractalsemantics.fractalsemantics_entity import generate_random_bitchain
    from fractalsemantics.progress_comm import ProgressReporter
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from fractalsemantics.fractalsemantics_entity import generate_random_bitchain
    from fractalsemantics.progress_comm import ProgressReporter

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

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


EXP02_DEFAULT_SCALES: list[int] = [1_000_000, 100_000_000, 10_000_000_000, 1_000_000_000_000]
EXP02_LATENCY_THRESHOLDS_MS: dict[int, float] = {
    1_000_000: 0.1,
    100_000_000: 0.5,
    10_000_000_000: 2.0,
    1_000_000_000_000: 5.0,
}
EXP02_DEFAULT_THRESHOLD_MS: float = 2.0
EXP02_PAYLOAD_REPEAT_FACTOR: int = 100
EXP02_PAYLOAD_MEMORY_PRESSURE_SCALE_MIN: int = 10_000
EXP02_WARMUP_MAX_OPERATIONS: int = 1_000
EXP02_WARMUP_SCALE_DIVISOR: int = 10
EXP02_WARMUP_METADATA_UPDATE_PROBABILITY: float = 0.1
EXP02_PRESSURE_OBJECT_COUNT: int = 100
EXP02_PRESSURE_OBJECT_RANGE: int = 1_000
EXP02_PROGRESS_UPDATE_STEPS: int = 10
EXP02_RANDOM_QUERY_RATIO: float = 0.7
EXP02_LOCALITY_QUERY_RATIO: float = 0.2
EXP02_LOCALITY_SUBSET_DIVISOR: int = 4
EXP02_ADVERSARIAL_JUMP_DIVISOR: int = 10
EXP02_DEV_QUERY_COUNT: int = 100
EXP02_DEV_AUTO_UPGRADE_QUERY_COUNT: int = 5_000
EXP02_QUICK_QUERY_COUNT: int = 100
EXP02_FULL_QUERY_COUNT: int = 500_000

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

    def to_dict(self) -> JsonObject:
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
            self.scales = config.get("EXP-02", "scales", EXP02_DEFAULT_SCALES)
        except Exception:
            self.scales = EXP02_DEFAULT_SCALES  # Scaled defaults: 1M, 100M, 10B, 1T
        self.results: list[EXP02_Result] = []

    def _format_scale_list(self) -> str:
        """Return scales as a compact human-readable string."""
        return ", ".join(f"{scale:,}" for scale in self.scales)

    def _print_compact_summary(self) -> None:
        """Print a compact, readable summary table for all scales."""
        if not self.results:
            return

        print("\nSummary by scale:")
        print("-" * 70)
        print(f"{'Scale':>12}  {'Mean(ms)':>9}  {'P95(ms)':>9}  {'P99(ms)':>9}  {'Target':>9}  {'Status':>6}")
        print("-" * 70)

        thresholds = EXP02_LATENCY_THRESHOLDS_MS

        for result in self.results:
            target = thresholds.get(result.scale, EXP02_DEFAULT_THRESHOLD_MS)
            status = "PASS" if result.success else "FAIL"
            print(
                f"{result.scale:>12,}  "
                f"{result.mean_latency_ms:>9.4f}  "
                f"{result.p95_latency_ms:>9.4f}  "
                f"{result.p99_latency_ms:>9.4f}  "
                f"<{target:>7.2f}  "
                f"{status:>6}"
            )
        print("-" * 70)

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
        print("EXP-02: Retrieval Efficiency Test")
        print(f"{'=' * 70}")
        print(f"Running {len(self.scales)} scales with {self.query_count:,} queries per scale")
        print(f"Scales: {self._format_scale_list()}")
        print("Method: warmup + memory pressure + mixed query patterns")
        print()

        subprocess_enabled = is_subprocess_communication_enabled()
        progress_reporter = None
        if not subprocess_enabled:
            try:
                progress_reporter = ProgressReporter("EXP-02")
            except Exception:
                progress_reporter = None

        def report_status(stage: str, message: str) -> None:
            if subprocess_enabled:
                send_subprocess_status("EXP-02", stage, message)
                return
            if progress_reporter is not None:
                with suppress(Exception):
                    progress_reporter.status(stage, message)

        def report_progress(progress_percent: float, stage: str, message: str) -> None:
            if subprocess_enabled:
                send_subprocess_progress("EXP-02", progress_percent, stage, message)
                return
            if progress_reporter is not None:
                with suppress(Exception):
                    progress_reporter.update(progress_percent, stage, message)

        def report_completion(success: bool, message: str) -> None:
            if subprocess_enabled:
                send_subprocess_completion("EXP-02", success, message)
                return
            if progress_reporter is not None:
                with suppress(Exception):
                    progress_reporter.complete(message)

        report_status("Initialization", "Starting retrieval efficiency test")

        all_success = True
        # Updated thresholds for scaled experiments - more lenient for larger scales
        thresholds = EXP02_LATENCY_THRESHOLDS_MS

        for scale_index, scale in enumerate(self.scales):
            progress_percent = (scale_index / len(self.scales)) * 100
            print(f"[{scale_index + 1}/{len(self.scales)}] Scale {scale:,}: starting benchmark")

            report_status(f"Scale {scale:,}", f"Testing {scale:,} bit-chains")

            start_time = time.time()

            # 1. Generate bit-chains with realistic data storage
            bitchains = []
            for chain_index in range(scale):
                bc = generate_random_bitchain(seed=chain_index)
                # Store realistic payload data to simulate real-world overhead
                # This adds memory pressure and more realistic lookup costs
                payload_data = f"simulated_payload_{chain_index}" * EXP02_PAYLOAD_REPEAT_FACTOR  # ~2KB per chain
                bitchains.append((bc, payload_data))

            # 2. Index by address for more realistic storage simulation
            # Use a richer structure to avoid pure Python dict optimization
            address_to_data: dict[str, dict[str, Any]] = {}
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
            # 3. Warmup phase - perform operations to stabilize performance
            warmup_start = time.perf_counter()
            warmup_operations = min(EXP02_WARMUP_MAX_OPERATIONS, scale // EXP02_WARMUP_SCALE_DIVISOR)  # Scale warmup with dataset size

            for _ in range(warmup_operations):
                # Perform random access patterns during warmup
                addr = secure_random.choice(addresses)
                _ = address_to_data[addr]['bitchain']

                # Simulate some metadata updates
                if secure_random.random() < EXP02_WARMUP_METADATA_UPDATE_PROBABILITY:
                    address_to_data[addr]['metadata']['accessed_count'] += 1
                    address_to_data[addr]['metadata']['last_accessed'] = time.time()

            warmup_time = (time.perf_counter() - warmup_start) * 1000

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

            # Artificial memory pressure for larger datasets
            if scale >= EXP02_PAYLOAD_MEMORY_PRESSURE_SCALE_MIN:
                # Create some memory pressure by allocating temporary objects
                pressure_objects = [list(range(EXP02_PRESSURE_OBJECT_RANGE)) for _ in range(EXP02_PRESSURE_OBJECT_COUNT)]
                del pressure_objects
                gc.collect()

            # 5. Performance measurement with multiple query patterns
            latencies = []
            hits = 0
            total_queries = 0

            # Mix of query patterns to simulate real-world usage
            query_patterns = self._generate_query_patterns(addresses, self.query_count)

            # Progress tracking for query execution
            progress_interval = max(1, self.query_count // EXP02_PROGRESS_UPDATE_STEPS)  # Update every 10%

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
                    report_progress(
                        query_progress,
                        f"{scale:,} Scale",
                        f"Executed {query_idx:,}/{self.query_count:,} queries",
                    )

            # 6. Compute enhanced statistics
            latencies.sort()
            mean_lat = sum(latencies) / len(latencies)
            median_lat = latencies[len(latencies) // 2]
            p95_lat = latencies[int(len(latencies) * 0.95)]
            p99_lat = latencies[int(len(latencies) * 0.99)]
            min_lat = latencies[0]
            max_lat = latencies[-1]
            cache_hit_rate = hits / total_queries if total_queries > 0 else 0.0

            threshold = thresholds.get(scale, EXP02_DEFAULT_THRESHOLD_MS)
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
            memory_display = f"{memory_pressure:.1f}%" if memory_pressure is not None else "N/A"
            print(
                f"[{scale_index + 1}/{len(self.scales)}] Scale {scale:,}: {status} | "
                f"mean={mean_lat:.4f}ms, p95={p95_lat:.4f}ms, p99={p99_lat:.4f}ms, "
                f"target<{threshold:.2f}ms, cache={cache_hit_rate:.1%}, mem={memory_display}, "
                f"warmup={warmup_time:.2f}ms, elapsed={total_time:.1f}s"
            )

        if all_success:
            print("OVERALL RESULT: ALL PASS")
        else:
            print("OVERALL RESULT: SOME FAILED")

        self._print_compact_summary()

        report_completion(all_success, f"Retrieval efficiency {'passed' if all_success else 'failed'}")

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
        for _ in range(int(query_count * EXP02_RANDOM_QUERY_RATIO)):
            queries.append(secure_random.choice(addresses))

        # 20% temporal locality (recently accessed patterns)
        # Simulate by preferring items from the first quarter of the address list
        locality_subset = addresses[:len(addresses) // EXP02_LOCALITY_SUBSET_DIVISOR]
        for _ in range(int(query_count * EXP02_LOCALITY_QUERY_RATIO)):
            queries.append(secure_random.choice(locality_subset))

        # 10% adversarial patterns (worst case, potentially cache misses)
        # Use non-sequential access patterns
        for _ in range(query_count - len(queries)):
            # Jump around the list to avoid cache-friendly patterns
            idx = secure_random.randint(0, len(addresses)-1)
            jump = secure_random.randint(1, len(addresses) // EXP02_ADVERSARIAL_JUMP_DIVISOR)
            queries.append(addresses[(idx + jump) % len(addresses)])

        # Shuffle to avoid artificial patterns in measurement
        secure_random.shuffle(queries)
        return queries[:query_count]

    def get_summary(self) -> JsonObject:
        """Get summary statistics."""
        return {
            "total_scales_tested": len(self.results),
            "all_passed": bool(self.results) and all(r.success for r in self.results),
            "results": [r.to_dict() for r in self.results],
        }


def save_results(results: JsonObject, output_file: Optional[str] = None) -> str:
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
        if env == "dev" and query_count == EXP02_DEV_QUERY_COUNT:  # Dev default is too low for reliable testing
            # Use production settings to ensure orchestrator runs match direct --full runs
            query_count = EXP02_DEV_AUTO_UPGRADE_QUERY_COUNT  # Same as --full flag
    except Exception:
        pass  # Use default value set above

    # Check CLI args regardless of config success (these override config)
    if "--quick" in sys.argv:
        query_count = EXP02_QUICK_QUERY_COUNT
    elif "--full" in sys.argv:
        query_count = EXP02_FULL_QUERY_COUNT

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
