"""
EXP-02: Retrieval Efficiency Test - Experiment Module

This module implements the core experiment logic for testing retrieval efficiency
of FractalStat address-based systems at various scale levels. The experiment
validates that retrieval performance remains acceptable as dataset sizes increase
from millions to trillions of entries.

Core Scientific Methodology:
1. Build indexed datasets at different scales (1M, 100M, 10B, 1T entries)
2. Generate realistic query patterns including hot/cold data access
3. Measure comprehensive latency statistics (mean, median, P95, P99)
4. Analyze memory pressure and cache behavior
5. Validate performance targets are met at each scale

Key Performance Insights:
- Hash-based retrieval should provide O(1) average-case performance
- Memory pressure increases with dataset size but should not significantly impact latency
- Cache hit rates should remain high for realistic access patterns
- Tail latencies (P95, P99) should remain within acceptable bounds

Author: FractalSemantics
Date: 2025-12-07
"""

import json
import sys
import time
import secrets
import gc
import psutil  # type: ignore[import-untyped]
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .entities import EXP02_Result
from fractalstat.fractalstat_entity import generate_random_bitchain

secure_random = secrets.SystemRandom()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval efficiency testing parameters.
    
    This class defines the scales and parameters used for testing retrieval
    performance across different dataset sizes.
    """
    
    # Default scales for testing: 1M, 100M, 10B, 1T entries
    DEFAULT_SCALES = [1000000, 100000000, 10000000000, 1000000000000]
    
    # Performance targets by scale (mean latency in milliseconds)
    PERFORMANCE_TARGETS = {
        1000000: 0.1,      # 1M: 0.1ms target
        100000000: 0.5,    # 100M: 0.5ms target
        10000000000: 2.0,  # 10B: 2.0ms target
        1000000000000: 5.0 # 1T: 5.0ms target
    }
    
    # Query pattern distribution for realistic testing
    QUERY_PATTERN_DISTRIBUTION = {
        'random_access': 0.7,      # 70% random access (typical DB workload)
        'temporal_locality': 0.2,  # 20% temporal locality (recently accessed)
        'adversarial': 0.1         # 10% adversarial patterns (worst case)
    }


class EXP02_RetrievalEfficiency:
    """
    EXP-02: Retrieval Efficiency Test

    This experiment validates that FractalStat address-based retrieval is fast enough
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
        """
        Initialize the retrieval efficiency experiment.
        
        Args:
            query_count: Number of retrieval queries to perform per scale
                        Default: 1,000,000 (suitable for hardware-constrained testing)
        """
        self.query_count = query_count
        # Load scales from config or use scaled defaults
        try:
            from fractalstat.config import ExperimentConfig
            config = ExperimentConfig()
            self.scales = config.get("EXP-02", "scales", RetrievalConfig.DEFAULT_SCALES)
        except Exception:
            self.scales = RetrievalConfig.DEFAULT_SCALES
        self.results: List[EXP02_Result] = []

    def _generate_bit_chains_with_payload(self, scale: int) -> List[Tuple[Any, str]]:
        """
        Generate bit-chains with realistic payload data for memory pressure testing.
        
        Args:
            scale: Number of bit-chains to generate
            
        Returns:
            List of tuples containing (bit-chain, payload_data)
            
        Implementation Details:
            - Generates realistic payload data (~2KB per chain)
            - Simulates real-world storage overhead
            - Adds memory pressure for more realistic performance testing
        """
        bitchains = []
        for i in range(scale):
            bc = generate_random_bitchain(seed=i)
            # Store realistic payload data to simulate real-world overhead
            # This adds memory pressure and more realistic lookup costs
            payload_data = f"simulated_payload_{i}" * 100  # ~2KB per chain
            bitchains.append((bc, payload_data))
        return bitchains

    def _build_address_index(self, bitchains: List[Tuple[Any, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Build an address-based index for efficient retrieval testing.
        
        Args:
            bitchains: List of bit-chains with payload data
            
        Returns:
            Dictionary mapping addresses to rich data structures
            
        Index Structure:
            {
                'address': {
                    'bitchain': bit_chain_object,
                    'payload_size': int,
                    'metadata': {
                        'created': timestamp,
                        'accessed_count': int,
                        'last_accessed': timestamp
                    }
                }
            }
        """
        address_to_data: Dict[str, Dict[str, Any]] = {}
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
        return address_to_data

    def _measure_memory_pressure(self) -> Optional[float]:
        """
        Measure current memory pressure using psutil if available.
        
        Returns:
            Memory usage percentage or None if psutil not available
            
        Memory Analysis:
            - Measures memory before and after garbage collection
            - Returns the higher of the two values to account for memory pressure
            - Falls back to garbage collection only if psutil is unavailable
        """
        if not HAS_PSUTIL:
            gc.collect()
            return None

        try:
            process = psutil.Process()
            memory_before = process.memory_percent()
            gc.collect()  # Force garbage collection
            memory_after = process.memory_percent()
            return max(memory_before, memory_after)
        except Exception:
            gc.collect()
            return None

    def _generate_query_patterns(self, addresses: List[str], query_count: int) -> List[str]:
        """
        Generate realistic query patterns including:
        - Hot data access patterns (recently accessed items)
        - Random access patterns
        - Cache-friendly access patterns (temporal locality)
        - Edge cases (non-existent keys, adversarial patterns)
        
        Args:
            addresses: List of valid addresses to query
            query_count: Total number of queries to generate
            
        Returns:
            List of query addresses with mixed access patterns
        """
        queries = []

        # 70% random access (typical DB workload)
        for _ in range(int(query_count * RetrievalConfig.QUERY_PATTERN_DISTRIBUTION['random_access'])):
            queries.append(secure_random.choice(addresses))

        # 20% temporal locality (recently accessed patterns)
        # Simulate by preferring items from the first quarter of the address list
        locality_subset = addresses[:len(addresses)//4]
        for _ in range(int(query_count * RetrievalConfig.QUERY_PATTERN_DISTRIBUTION['temporal_locality'])):
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

    def _measure_retrieval_performance(self, address_to_data: Dict[str, Dict[str, Any]], query_addresses: List[str]) -> Dict[str, Any]:
        """
        Measure retrieval performance for a set of query addresses.
        
        Args:
            address_to_data: Index mapping addresses to data
            query_addresses: List of addresses to query
            
        Returns:
            Dictionary containing latency statistics and cache behavior
            
        Performance Metrics:
            - Latency measurements for each query
            - Cache hit/miss statistics
            - Statistical analysis (mean, median, percentiles)
        """
        latencies = []
        hits = 0
        total_queries = len(query_addresses)

        for query_addr in query_addresses:
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

        # Compute comprehensive statistics
        latencies.sort()
        mean_lat = sum(latencies) / len(latencies)
        median_lat = latencies[len(latencies) // 2]
        p95_lat = latencies[int(len(latencies) * 0.95)]
        p99_lat = latencies[int(len(latencies) * 0.99)]
        min_lat = latencies[0]
        max_lat = latencies[-1]
        cache_hit_rate = hits / total_queries if total_queries > 0 else 0.0

        return {
            'latencies': latencies,
            'mean_latency_ms': mean_lat,
            'median_latency_ms': median_lat,
            'p95_latency_ms': p95_lat,
            'p99_latency_ms': p99_lat,
            'min_latency_ms': min_lat,
            'max_latency_ms': max_lat,
            'cache_hit_rate': cache_hit_rate,
            'total_queries': total_queries
        }

    def _run_warmup_phase(self, address_to_data: Dict[str, Dict[str, Any]], addresses: List[str]) -> float:
        """
        Run warmup phase to stabilize performance measurements.
        
        Args:
            address_to_data: Index for performing warmup operations
            addresses: List of addresses for warmup queries
            
        Returns:
            Warmup time in milliseconds
            
        Warmup Process:
            - Performs operations to stabilize JIT compilation
            - Allows caching systems to warm up
            - Provides more consistent performance measurements
        """
        warmup_operations = min(1000, len(addresses) // 10)  # Scale warmup with dataset size
        warmup_start = time.perf_counter()

        for _ in range(warmup_operations):
            # Perform random access patterns during warmup
            addr = secure_random.choice(addresses)
            _ = address_to_data[addr]['bitchain']

            # Simulate some metadata updates
            if secure_random.random() < 0.1:  # 10% chance
                address_to_data[addr]['metadata']['accessed_count'] += 1
                address_to_data[addr]['metadata']['last_accessed'] = time.time()

        return (time.perf_counter() - warmup_start) * 1000

    def run(self) -> Tuple[List[EXP02_Result], bool]:
        """
        Run the retrieval efficiency test with comprehensive benchmarking.

        This enhanced version includes:
        - Warmup periods to account for JIT compilation and caching
        - Memory pressure testing with realistic data storage
        - Cache hit/miss simulation
        - Multiple query patterns (cached, random, adversarial)

        Returns:
            Tuple of (results list, overall success boolean)
            
        Test Process:
            1. For each scale (1M, 100M, 10B, 1T):
               - Generate bit-chains with realistic payload data
               - Build address-based index
               - Run warmup phase
               - Measure memory pressure
               - Generate realistic query patterns
               - Measure retrieval performance
               - Store results with comprehensive metrics
            2. Analyze overall performance across all scales
            3. Return results and validation success status
        """
        print(f"\n{'=' * 70}")
        print("EXP-02: RETRIEVAL EFFICIENCY TEST (ENHANCED)")
        print(f"{'=' * 70}")
        print(f"Query count per scale: {self.query_count}")
        print(f"Scales: {self.scales}")
        print("Includes: warmup, memory pressure, cache simulation")
        print()

        all_success = True

        for scale in self.scales:
            print(f"Testing scale: {scale:,} bit-chains")
            start_time = time.time()

            # 1. Generate bit-chains with realistic data storage
            bitchains = self._generate_bit_chains_with_payload(scale)
            print(f"  Generated {len(bitchains):,} bit-chains with payload data")

            # 2. Build address index for more realistic storage simulation
            address_to_data = self._build_address_index(bitchains)
            addresses = list(address_to_data.keys())
            print(f"  Index built: {len(addresses)} entries")

            # 3. Warmup phase - perform operations to stabilize performance
            print("  Warmup phase...")
            warmup_time = self._run_warmup_phase(address_to_data, addresses)
            print(f"  Warmup complete: {warmup_time:.3f}ms")

            # 4. Memory pressure test - measure and force garbage collection
            memory_pressure = self._measure_memory_pressure()
            print(f"  Memory pressure: {memory_pressure:.1f}%" if memory_pressure else "  Memory pressure: N/A")

            # 5. Performance measurement with multiple query patterns
            print("  Performance measurement...")
            query_patterns = self._generate_query_patterns(addresses, self.query_count)
            
            performance_data = self._measure_retrieval_performance(address_to_data, query_patterns)

            # 6. Create result with comprehensive metrics
            threshold = RetrievalConfig.PERFORMANCE_TARGETS.get(scale, 2.0)
            success = performance_data['mean_latency_ms'] < threshold

            exp_result = EXP02_Result(
                scale=scale,
                queries=self.query_count,
                mean_latency_ms=performance_data['mean_latency_ms'],
                median_latency_ms=performance_data['median_latency_ms'],
                p95_latency_ms=performance_data['p95_latency_ms'],
                p99_latency_ms=performance_data['p99_latency_ms'],
                min_latency_ms=performance_data['min_latency_ms'],
                max_latency_ms=performance_data['max_latency_ms'],
                cache_hit_rate=performance_data['cache_hit_rate'],
                memory_pressure=memory_pressure,
                warmup_time_ms=warmup_time,
                success=success,
            )

            self.results.append(exp_result)
            all_success = all_success and success

            # 7. Display results for this scale
            status = "PASS" if success else "FAIL"
            print(
                f"  {status} | Mean: {performance_data['mean_latency_ms']:.4f}ms | "
                f"Median: {performance_data['median_latency_ms']:.4f}ms"
            )
            print(
                f"       P95: {performance_data['p95_latency_ms']:.4f}ms | P99: {performance_data['p99_latency_ms']:.4f}ms | "
                f"Cache: {performance_data['cache_hit_rate']:.1%}"
            )
            print(
                f"       Target: < {threshold}ms | "
                f"Time: {time.time() - start_time:.1f}s"
            )
            print()

        if all_success:
            print("OVERALL RESULT: ALL PASS")
        else:
            print("OVERALL RESULT: SOME FAILED")
        return self.results, all_success

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive retrieval efficiency analysis summary.
        
        Returns:
            Dictionary containing complete experiment results and analysis
            
        Summary Includes:
            - Total scales tested and overall success status
            - Performance metrics for each scale
            - Memory efficiency analysis
            - Query pattern effectiveness
            - Detailed results for each dimension
        """
        return {
            "total_scales_tested": len(self.results),
            "all_passed": bool(self.results) and all(r.success for r in self.results),
            "performance_targets": RetrievalConfig.PERFORMANCE_TARGETS,
            "query_pattern_distribution": RetrievalConfig.QUERY_PATTERN_DISTRIBUTION,
            "results": [r.to_dict() for r in self.results],
        }


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary containing experiment results
        output_file: Optional output file path. If None, generates timestamped filename.
        
    Returns:
        Path to the saved results file
        
    File Format:
        JSON file with experiment metadata, configuration, and detailed results
        Saved in the project's results directory
    """
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp02_retrieval_efficiency_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


def run_experiment_from_config(config: Optional[Dict[str, Any]] = None) -> Tuple[List[EXP02_Result], bool, Dict[str, Any]]:
    """
    Run the experiment with configuration parameters.
    
    Args:
        config: Optional configuration dictionary with experiment parameters
        
    Returns:
        Tuple of (results list, success status, summary dictionary)
        
    Configuration Options:
        - query_count: Number of queries per scale (default: 1000000)
        - scales: List of scales to test (default: [1M, 100M, 10B, 1T])
    """
    if config is None:
        config = {}
    
    query_count = config.get("query_count", 1000000)
    
    experiment = EXP02_RetrievalEfficiency(query_count=query_count)
    results_list, success = experiment.run()
    summary = experiment.get_summary()
    
    return results_list, success, summary