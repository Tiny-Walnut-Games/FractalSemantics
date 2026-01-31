"""
EXP-04: FractalStat Fractal Scaling Test - Experiment Logic

This module contains the core experiment logic for testing fractal scaling
properties of FractalStat addressing across different data scales.
"""

from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
import time
import statistics
import math
from collections import defaultdict

from .entities import (
    ScaleTestConfig,
    ScaleTestResults,
    FractalScalingResults,
    generate_random_bitchain,
    secure_random,
)


def run_scale_test(config: ScaleTestConfig) -> ScaleTestResults:
    """
    Run EXP-01 (uniqueness) + EXP-02 (retrieval) at a single scale.

    Returns:
        ScaleTestResults with collision and latency metrics
    """
    start_time = time.time()

    # Step 1: Generate bit-chains
    print(f"  Generating {config.scale} bit-chains...", end="", flush=True)
    bitchains: List[BitChain] = []
    for i in range(config.scale):
        bitchains.append(generate_random_bitchain())
    print(" OK")

    # Step 2: Compute addresses and check for collisions (EXP-01)
    print("  Computing addresses (EXP-01)...", end="", flush=True)
    address_map: Dict[str, int] = defaultdict(int)
    addresses = []

    for bc in bitchains:
        addr = bc.compute_address()
        addresses.append(addr)
        address_map[addr] += 1

    unique_addresses = len(address_map)
    collision_groups = sum(1 for count in address_map.values() if count > 1)
    collision_count = sum(count - 1 for count in address_map.values() if count > 1)
    collision_rate = collision_count / config.scale if config.scale > 0 else 0.0
    print(
        f" OK ({unique_addresses} unique, {collision_groups} collision groups, {
            collision_count
        } total collisions)"
    )

    # Step 3: Build retrieval index
    print("  Building retrieval index...", end="", flush=True)
    address_to_bitchain = {addr: bc for bc, addr in zip(bitchains, addresses)}
    print(" OK")

    # Step 4: Test retrieval performance (EXP-02)
    print(
        f"  Testing retrieval ({config.num_retrievals} queries)...",
        end="",
        flush=True,
    )
    retrieval_times = []

    for _ in range(config.num_retrievals):
        idx = secure_random.randint(0, len(addresses) - 1)
        target_addr = addresses[idx]

        # Measure lookup time
        start_lookup = time.perf_counter()
        result = address_to_bitchain.get(target_addr)
        end_lookup = time.perf_counter()

        if result is None:
            raise RuntimeError(f"Address lookup failed for {target_addr}")

        retrieval_times.append((end_lookup - start_lookup) * 1000)  # Convert to ms

    print(" OK")

    # Step 5: Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time

    retrieval_mean = statistics.mean(retrieval_times)
    retrieval_median = statistics.median(retrieval_times)
    retrieval_p95 = sorted(retrieval_times)[int(len(retrieval_times) * 0.95)]
    retrieval_p99 = sorted(retrieval_times)[int(len(retrieval_times) * 0.99)]

    # Prevent division by zero for very fast tests
    addresses_per_second = config.scale / max(total_time, 0.001)

    return ScaleTestResults(
        scale=config.scale,
        num_bitchains=config.scale,
        num_addresses=len(addresses),
        unique_addresses=unique_addresses,
        collision_count=collision_count,
        collision_rate=collision_rate,
        num_retrievals=config.num_retrievals,
        retrieval_times_ms=retrieval_times,
        retrieval_mean_ms=retrieval_mean,
        retrieval_median_ms=retrieval_median,
        retrieval_p95_ms=retrieval_p95,
        retrieval_p99_ms=retrieval_p99,
        total_time_seconds=total_time,
        addresses_per_second=addresses_per_second,
    )


def analyze_degradation(
    results: List[ScaleTestResults],
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Analyze whether the system maintains fractal properties.

    Returns:
        (collision_analysis, retrieval_analysis, is_fractal)
    """
    is_fractal = True

    # Check collisions
    collision_msg = ""
    max_collision_rate = max(r.collision_rate for r in results)

    if max_collision_rate > 0.0:
        is_fractal = False
        collision_msg = f"COLLISION DETECTED: Max rate {max_collision_rate * 100:.2f}%"
    else:
        collision_msg = "OK Zero collisions at all scales"

    # Check retrieval degradation
    retrieval_msg = ""
    retrieval_means = [r.retrieval_mean_ms for r in results]

    # Check if retrieval time is increasing linearly with scale (bad) vs
    # logarithmically (good)
    scales = [r.scale for r in results]
    if len(results) > 1:
        # Simple degradation check: is the retrieval time growing faster than
        # O(log n)?
        worst_case_ratio = max(retrieval_means) / min(retrieval_means)
        scale_ratio = max(scales) / min(scales)

        # If retrieval grows faster than log scale ratio, it's degrading too
        # fast
        expected_log_ratio = math.log(scale_ratio)

        if worst_case_ratio > expected_log_ratio * 2:
            is_fractal = False
            retrieval_msg = f"DEGRADATION WARNING: Latency ratio {
                worst_case_ratio:.2f}x > expected {expected_log_ratio:.2f}x"
        else:
            retrieval_msg = f"OK Retrieval latency scales logarithmically ({
                worst_case_ratio:.2f}x for {scale_ratio:.0f}x scale)"

    return collision_msg, retrieval_msg, is_fractal


def run_fractal_scaling_test(quick_mode: bool = True) -> FractalScalingResults:
    """
    Run EXP-04: Fractal Scaling Test

    Args:
        quick_mode: If True, test 1K, 10K, 100K. If False, also test 1M.

    Returns:
        Complete results object
    """
    # Load experiment configuration
    try:
        from fractalstat.config import ExperimentConfig

        config = ExperimentConfig()
        quick_mode = config.get("EXP-04", "quick_mode", quick_mode)
        scales = config.get("EXP-04", "scales", [1_000, 10_000, 100_000])
        num_retrievals = config.get("EXP-04", "num_retrievals", 1000)
    except Exception:
        # Fallback to default parameters if config not available
        scales = [1_000, 10_000, 100_000]
        if not quick_mode:
            scales.append(1_000_000)
        num_retrievals = 1000

    start_time = datetime.now(timezone.utc).isoformat()
    overall_start = time.time()

    print("\n" + "=" * 70)
    print("EXP-04: FractalStat FRACTAL SCALING TEST")
    print("=" * 70)
    print(f"Mode: {'Quick' if quick_mode else 'Full'} (scales: {scales})")
    print()

    scale_results = []

    for scale in scales:
        print(f"SCALE: {scale:,} bit-chains")
        print("-" * 70)

        scale_config = ScaleTestConfig(
            scale=scale,
            num_retrievals=num_retrievals,
            timeout_seconds=300,
        )

        try:
            result = run_scale_test(scale_config)
            scale_results.append(result)

            # Print summary for this scale
            print(f"  RESULT: {result.num_addresses} unique addresses")
            print(
                f"          Collisions: {result.collision_count} ({
                    result.collision_rate * 100:.2f}%)"
            )
            print(
                f"          Retrieval: mean={result.retrieval_mean_ms:.6f}ms, p95={
                    result.retrieval_p95_ms:.6f}ms"
            )
            print(f"          Throughput: {result.addresses_per_second:,.0f} addr/sec")
            print(f"          Valid: {'YES' if result.is_valid() else 'NO'}")
            print()

        except Exception as e:
            print(f"  FAILED: {e}")
            print()
            raise

    # Analyze degradation
    collision_analysis, retrieval_analysis, is_fractal = analyze_degradation(
        scale_results
    )

    overall_end = time.time()
    end_time = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("DEGRADATION ANALYSIS")
    print("=" * 70)
    print(f"Collision: {collision_analysis}")
    print(f"Retrieval: {retrieval_analysis}")
    print(f"Is Fractal: {'YES' if is_fractal else 'NO'}")
    print()

    results = FractalScalingResults(
        start_time=start_time,
        end_time=end_time,
        total_duration_seconds=(overall_end - overall_start),
        scale_results=scale_results,
        collision_degradation=collision_analysis,
        retrieval_degradation=retrieval_analysis,
        is_fractal=is_fractal,
    )

    return results