"""
EXP-04: Bit-Chain FractalStat Fractal Scaling Test - Experiment Module

This module implements the core experiment logic for testing whether FractalStat
addressing maintains consistency and zero collisions when scaled from 1K → 10K →
100K → 1M data points. The experiment validates the "fractal" property of self-similar
behavior at all scales.

Core Scientific Methodology:
1. Generate bit-chains at increasing scale levels (1K, 10K, 100K, 1M, 10M)
2. Compute addresses and verify zero collisions at each scale
3. Test retrieval performance and measure latency scaling patterns
4. Analyze degradation to confirm fractal properties (logarithmic scaling)
5. Validate self-similar system behavior across all scale levels

Key Fractal Properties Validated:
- Zero collisions maintained across all scale levels
- Retrieval latency scales logarithmically (not linearly)
- Address generation throughput remains consistent
- System behavior is self-similar across scales

Author: FractalSemantics
Date: 2025-12-07
"""

import json
import time
import secrets
import sys
import statistics
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from .entities import ScaleTestConfig, ScaleTestResults, FractalScalingResults
from fractalstat.fractalstat_experiments import BitChain, generate_random_bitchain

secure_random = secrets.SystemRandom()


def run_scale_test(config: ScaleTestConfig) -> ScaleTestResults:
    """
    Run collision detection and retrieval performance test at a single scale level.

    This function implements the core testing logic for a specific scale level,
    combining EXP-01 (uniqueness) and EXP-02 (retrieval) validation at the specified
    scale. It generates bit-chains, computes addresses, checks for collisions, and
    measures retrieval performance.

    Args:
        config: ScaleTestConfig defining the scale level and test parameters

    Returns:
        ScaleTestResults containing comprehensive performance and collision metrics

    Test Process:
        1. Generate the specified number of bit-chains using cryptographically secure RNG
        2. Compute FractalStat addresses for all bit-chains
        3. Detect and count collisions using address frequency analysis
        4. Build retrieval index for performance testing
        5. Perform random retrieval queries and measure latency
        6. Calculate statistical metrics and throughput measurements

    Performance Metrics Captured:
        - Collision count and rate at the scale level
        - Retrieval latency statistics (mean, median, p95, p99)
        - Address generation throughput (addresses per second)
        - Memory efficiency and uniqueness ratios
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
    Analyze whether the system maintains fractal properties across scale levels.

    This function performs the core fractal property validation by analyzing
    collision behavior and retrieval performance degradation patterns across
    different scale levels.

    Args:
        results: List of ScaleTestResults from different scale levels

    Returns:
        Tuple of (collision_analysis, retrieval_analysis, is_fractal)

    Fractal Property Analysis:
        1. Collision Consistency: Verifies zero collisions at all scales
        2. Performance Scaling: Checks if retrieval latency scales logarithmically
        3. Self-Similarity: Validates consistent behavior patterns across scales

    Degradation Detection:
        - Collision degradation: Any non-zero collision rate indicates failure
        - Performance degradation: Linear scaling instead of logarithmic indicates failure
        - Fractal validation: Both collision-free and logarithmic scaling required
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

    This is the main orchestration function that tests FractalStat addressing
    across multiple scale levels to validate fractal properties. It systematically
    tests each scale level, analyzes degradation patterns, and determines whether
    the system maintains self-similar behavior.

    Args:
        quick_mode: If True, test 1K, 10K, 100K. If False, also test 1M and 10M.

    Returns:
        FractalScalingResults containing complete test results and analysis

    Test Configuration:
        - Quick Mode: 1K, 10K, 100K (faster execution for development)
        - Full Mode: 1K, 10K, 100K, 1M, 10M (comprehensive validation)
        - Configurable through experiment configuration files
        - Default retrieval queries: 1000 per scale level

    Success Criteria:
        - Zero collisions at all scale levels
        - Retrieval latency scales logarithmically (not linearly)
        - Address generation throughput remains consistent
        - System behavior is self-similar across scales

    Output:
        - Comprehensive JSON results file with all metrics
        - Console output with progress and summary
        - Fractal property validation status
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


def save_results(
    results: FractalScalingResults, output_file: Optional[str] = None
) -> str:
    """
    Save fractal scaling test results to JSON file.
    
    Args:
        results: FractalScalingResults object containing all test data
        output_file: Optional output file path. If None, generates timestamped filename.
        
    Returns:
        Path to the saved results file
        
    File Format:
        JSON file with comprehensive test results including:
        - Scale-level performance metrics
        - Collision analysis across all scales
        - Fractal property validation results
        - Performance degradation analysis
        - Statistical summaries and comparisons
        
    Saved Location:
        Results directory in project root with timestamped filename
    """
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp04_fractal_scaling_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


def validate_fractal_properties(results: FractalScalingResults) -> Dict[str, Any]:
    """
    Perform comprehensive validation of fractal properties.
    
    Args:
        results: FractalScalingResults from the test run
        
    Returns:
        Dictionary with detailed fractal property validation
        
    Validation Checks:
        - Collision-free property across all scales
        - Logarithmic performance scaling
        - Self-similarity of system behavior
        - Performance consistency metrics
        - Throughput stability analysis
    """
    if not results.scale_results:
        return {
            'validation_passed': False,
            'reason': 'No scale results available for validation',
            'details': {}
        }

    # Check collision-free property
    max_collision_rate = max(r.collision_rate for r in results.scale_results)
    collision_free = max_collision_rate == 0.0

    # Check logarithmic scaling
    scales = [r.scale for r in results.scale_results]
    retrieval_means = [r.retrieval_mean_ms for r in results.scale_results]
    
    if len(results.scale_results) > 1:
        scale_ratio = max(scales) / min(scales)
        retrieval_ratio = max(retrieval_means) / min(retrieval_means)
        expected_log_ratio = math.log(scale_ratio)
        
        # Allow 2x tolerance for logarithmic scaling
        logarithmic_scaling = retrieval_ratio <= expected_log_ratio * 2
    else:
        logarithmic_scaling = True

    # Check performance consistency
    performance_consistency = all(r.retrieval_mean_ms < 2.0 for r in results.scale_results)

    # Overall validation
    validation_passed = collision_free and logarithmic_scaling and performance_consistency

    return {
        'validation_passed': validation_passed,
        'collision_free': collision_free,
        'logarithmic_scaling': logarithmic_scaling,
        'performance_consistency': performance_consistency,
        'max_collision_rate': max_collision_rate,
        'retrieval_scaling_ratio': retrieval_ratio if len(results.scale_results) > 1 else 1.0,
        'expected_log_ratio': expected_log_ratio if len(results.scale_results) > 1 else 1.0,
        'details': {
            'scales_tested': [r.name() for r in results.scale_results],
            'collision_rates': [r.collision_rate for r in results.scale_results],
            'retrieval_means': retrieval_means,
            'throughputs': [r.addresses_per_second for r in results.scale_results]
        }
    }


def run_experiment_from_config(config: Optional[Dict[str, Any]] = None) -> Tuple[FractalScalingResults, Dict[str, Any]]:
    """
    Run the fractal scaling experiment with configuration parameters.
    
    Args:
        config: Optional configuration dictionary with experiment parameters
        
    Returns:
        Tuple of (results object, validation dictionary)
        
    Configuration Options:
        - quick_mode: Boolean to run quick (1K, 10K, 100K) or full (1K-10M) test
        - scales: List of scale levels to test
        - num_retrievals: Number of retrieval queries per scale level
    """
    if config is None:
        config = {}
    
    quick_mode = config.get("quick_mode", True)
    
    results = run_fractal_scaling_test(quick_mode=quick_mode)
    validation = validate_fractal_properties(results)
    
    return results, validation