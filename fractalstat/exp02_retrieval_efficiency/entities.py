"""
EXP-02: Retrieval Efficiency Test - Entities Module

This module defines the core data structures and entities used in the retrieval
efficiency experiment. These entities capture performance metrics, latency
measurements, and system behavior under different scale conditions.

The entities are designed to capture:
- Latency percentiles and statistical distributions
- Memory pressure and cache behavior
- Performance targets and success criteria
- Query pattern analysis and efficiency metrics

Author: FractalSemantics
Date: 2025-12-07
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


@dataclass
class EXP02_Result:
    """
    Results from EXP-02 retrieval efficiency test.
    
    This dataclass captures comprehensive performance metrics for address-based
    retrieval operations across different dataset scales.
    
    Attributes:
        scale: Number of bit-chains in the dataset being tested
        queries: Number of retrieval queries performed
        mean_latency_ms: Average retrieval latency in milliseconds
        median_latency_ms: Median retrieval latency in milliseconds
        p95_latency_ms: 95th percentile retrieval latency in milliseconds
        p99_latency_ms: 99th percentile retrieval latency in milliseconds
        min_latency_ms: Minimum retrieval latency in milliseconds
        max_latency_ms: Maximum retrieval latency in milliseconds
        cache_hit_rate: Percentage of queries that hit cached data
        memory_pressure: Memory usage as percentage (optional)
        warmup_time_ms: Time spent in warmup phase in milliseconds
        success: Whether performance targets were met
    
    Performance Targets:
        - 1M bit-chains: Mean latency < 0.1ms
        - 100M bit-chains: Mean latency < 0.5ms
        - 10B bit-chains: Mean latency < 2.0ms
        - 1T bit-chains: Mean latency < 5.0ms
    """
    
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the result suitable for JSON export
            
        Usage:
            result_dict = result.to_dict()
            json.dump(result_dict, file)
        """
        return asdict(self)

    def __str__(self) -> str:
        """
        String representation of the retrieval efficiency test result.
        
        Returns:
            Human-readable summary of the performance test results
            
        Example:
            "1M scale: 1000 queries, mean=0.123ms, P95=0.456ms, success=True"
        """
        return (
            f"{self.scale:,} scale: {self.queries} queries, "
            f"mean={self.mean_latency_ms:.3f}ms, "
            f"P95={self.p95_latency_ms:.3f}ms, "
            f"success={self.success}"
        )

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get a summary of key performance metrics.
        
        Returns:
            Dictionary containing key latency and efficiency metrics
            
        Usage:
            summary = result.get_performance_summary()
            print(f"Mean: {summary['mean_latency_ms']}ms")
        """
        return {
            'mean_latency_ms': self.mean_latency_ms,
            'median_latency_ms': self.median_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'cache_hit_rate': self.cache_hit_rate,
            'warmup_time_ms': self.warmup_time_ms
        }

    def get_latency_distribution(self) -> Dict[str, float]:
        """
        Get latency distribution metrics for analysis.
        
        Returns:
            Dictionary with latency percentiles and range information
            
        Analysis Usage:
            distribution = result.get_latency_distribution()
            # Analyze tail latency behavior
            tail_ratio = distribution['p99_latency_ms'] / distribution['mean_latency_ms']
        """
        return {
            'min_latency_ms': self.min_latency_ms,
            'mean_latency_ms': self.mean_latency_ms,
            'median_latency_ms': self.median_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'latency_range_ms': self.max_latency_ms - self.min_latency_ms,
            'p99_to_mean_ratio': self.p99_latency_ms / self.mean_latency_ms if self.mean_latency_ms > 0 else 0.0
        }

    def is_within_performance_target(self) -> bool:
        """
        Check if the result meets the performance target for its scale.
        
        Returns:
            True if performance targets are met, False otherwise
            
        Performance Targets:
            - 1M scale: < 0.1ms mean latency
            - 100M scale: < 0.5ms mean latency
            - 10B scale: < 2.0ms mean latency
            - 1T scale: < 5.0ms mean latency
        """
        # Define performance targets based on scale
        targets = {
            1000000: 0.1,      # 1M: 0.1ms target
            100000000: 0.5,    # 100M: 0.5ms target
            10000000000: 2.0,  # 10B: 2.0ms target
            1000000000000: 5.0 # 1T: 5.0ms target
        }
        
        target_latency = targets.get(self.scale, 2.0)  # Default to 2.0ms for unknown scales
        return self.mean_latency_ms < target_latency

    def get_memory_efficiency(self) -> Dict[str, Any]:
        """
        Get memory efficiency and pressure metrics.
        
        Returns:
            Dictionary with memory usage and efficiency information
            
        Memory Analysis:
            - memory_pressure: Current memory usage percentage
            - cache_efficiency: How well caching is working
            - memory_scale_ratio: Memory usage relative to dataset scale
        """
        return {
            'memory_pressure': self.memory_pressure,
            'cache_hit_rate': self.cache_hit_rate,
            'queries_per_ms': self.queries / self.mean_latency_ms if self.mean_latency_ms > 0 else 0.0,
            'memory_scale_ratio': self.memory_pressure / self.scale if self.memory_pressure and self.scale > 0 else None
        }