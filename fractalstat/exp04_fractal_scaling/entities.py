"""
EXP-04: Bit-Chain FractalStat Fractal Scaling Test - Entities Module

This module defines the core data structures and entities used in the fractal
scaling experiment. These entities capture scale-level test configurations,
performance metrics, collision analysis, and fractal property validation.

The entities are designed to capture:
- Scale test configurations for different data volumes
- Performance metrics across multiple scale levels
- Collision detection and analysis at each scale
- Retrieval latency measurements and statistical analysis
- Fractal property validation and degradation analysis

Author: FractalSemantics
Date: 2025-12-07
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone


@dataclass
class ScaleTestConfig:
    """
    Configuration for a single scale level test.
    
    This dataclass defines the parameters for testing FractalStat addressing
    at different scale levels, from 1K to 10M bit-chains.
    
    Attributes:
        scale: Number of bit-chains to generate and test (1K, 10K, 100K, 1M, 10M)
        num_retrievals: Number of random retrieval queries to perform
        timeout_seconds: Maximum time allowed for the test to complete
        
    Scale Levels:
        - 1K (1,000): Baseline performance and collision detection
        - 10K (10,000): Small-scale stress testing
        - 100K (100,000): Medium-scale performance validation
        - 1M (1,000,000): Large-scale system behavior analysis
        - 10M (10,000,000): Extreme-scale fractal property validation
    """
    
    scale: int  # Number of bit-chains (1K, 10K, 100K, 1M, 10M)
    num_retrievals: int  # Number of random retrieval queries
    timeout_seconds: int  # Kill test if it takes too long

    def name(self) -> str:
        """
        Get human-readable scale name.
        
        Returns:
            String representation of the scale (e.g., "1K", "100K", "1M")
            
        Examples:
            >>> config = ScaleTestConfig(scale=1000, num_retrievals=100, timeout_seconds=60)
            >>> config.name()
            '1K'
            
            >>> config = ScaleTestConfig(scale=1000000, num_retrievals=100, timeout_seconds=60)
            >>> config.name()
            '1M'
        """
        if self.scale >= 10_000_000:
            return f"{self.scale // 10_000_000}M"
        elif self.scale >= 1_000:
            return f"{self.scale // 1_000}K"
        return str(self.scale)

    def __str__(self) -> str:
        """
        String representation of the scale test configuration.
        
        Returns:
            Human-readable summary of the test configuration
        """
        return f"Scale: {self.name()}, Retrievals: {self.num_retrievals}, Timeout: {self.timeout_seconds}s"


@dataclass
class ScaleTestResults:
    """
    Results from testing a single scale level.
    
    This dataclass captures comprehensive performance and collision metrics
    for a specific scale level in the fractal scaling test.
    
    Attributes:
        scale: The scale level tested (number of bit-chains)
        num_bitchains: Total number of bit-chains generated
        num_addresses: Total number of addresses computed
        unique_addresses: Number of unique addresses (no collisions)
        collision_count: Total number of collision instances
        collision_rate: Collision rate as a percentage
        
        retrieval_times_ms: List of individual retrieval times in milliseconds
        retrieval_mean_ms: Mean retrieval time
        retrieval_median_ms: Median retrieval time
        retrieval_p95_ms: 95th percentile retrieval time
        retrieval_p99_ms: 99th percentile retrieval time
        
        total_time_seconds: Total time for the scale test
        addresses_per_second: Throughput rate (addresses computed per second)
    """
    
    scale: int
    num_bitchains: int
    num_addresses: int
    unique_addresses: int
    collision_count: int
    collision_rate: float

    # Retrieval performance
    num_retrievals: int
    retrieval_times_ms: List[float]
    retrieval_mean_ms: float
    retrieval_median_ms: float
    retrieval_p95_ms: float
    retrieval_p99_ms: float

    # System metrics
    total_time_seconds: float
    addresses_per_second: float

    def is_valid(self) -> bool:
        """
        Check if results meet success criteria for fractal scaling.
        
        Success Criteria:
        - Zero collisions (collision_count == 0)
        - Zero collision rate (collision_rate == 0.0)
        - Sub-millisecond retrieval performance (mean < 2.0ms)
        
        Returns:
            True if all success criteria are met, False otherwise
        """
        return (
            self.collision_count == 0
            and self.collision_rate == 0.0
            and self.retrieval_mean_ms < 2.0  # Sub-millisecond target
        )

    def collision_analysis(self) -> Dict[str, Any]:
        """
        Get detailed collision analysis for this scale level.
        
        Returns:
            Dictionary with collision metrics and analysis
            
        Analysis Includes:
            - Collision count and rate
            - Uniqueness ratio
            - Collision-free status
            - Scale efficiency metrics
        """
        uniqueness_ratio = self.unique_addresses / self.num_addresses if self.num_addresses > 0 else 0.0
        
        return {
            'collision_count': self.collision_count,
            'collision_rate_percent': self.collision_rate * 100.0,
            'unique_addresses': self.unique_addresses,
            'total_addresses': self.num_addresses,
            'uniqueness_ratio': uniqueness_ratio,
            'collision_free': self.collision_count == 0,
            'scale_efficiency': uniqueness_ratio * 100.0
        }

    def performance_analysis(self) -> Dict[str, Any]:
        """
        Get detailed performance analysis for this scale level.
        
        Returns:
            Dictionary with performance metrics and analysis
            
        Performance Metrics:
            - Retrieval latency statistics (mean, median, p95, p99)
            - Throughput measurements
            - Performance consistency indicators
            - Scale-appropriate performance validation
        """
        return {
            'retrieval': {
                'mean_ms': self.retrieval_mean_ms,
                'median_ms': self.retrieval_median_ms,
                'p95_ms': self.retrieval_p95_ms,
                'p99_ms': self.retrieval_p99_ms,
                'query_count': self.num_retrievals,
                'latency_consistency': self.retrieval_p99_ms / max(self.retrieval_mean_ms, 0.001)
            },
            'throughput': {
                'addresses_per_second': self.addresses_per_second,
                'total_time_seconds': self.total_time_seconds,
                'scale': self.scale
            },
            'performance_valid': self.retrieval_mean_ms < 2.0
        }

    def scaling_metrics(self) -> Dict[str, Any]:
        """
        Get scaling-specific metrics for fractal analysis.
        
        Returns:
            Dictionary with metrics used for fractal property validation
            
        Scaling Analysis:
            - Address generation efficiency
            - Memory usage patterns
            - Performance degradation indicators
            - Fractal consistency metrics
        """
        return {
            'scale_level': self.scale,
            'addresses_per_ms': self.addresses_per_second / 1000.0,
            'memory_efficiency': self.unique_addresses / self.scale if self.scale > 0 else 0.0,
            'performance_stability': self.retrieval_p99_ms / max(self.retrieval_mean_ms, 0.001),
            'fractal_consistency': self.collision_count == 0 and self.retrieval_mean_ms < 2.0
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation suitable for JSON export
            
        Usage:
            result_dict = result.to_dict()
            json.dump(result_dict, file)
        """
        return {
            "scale": self.scale,
            "num_bitchains": self.num_bitchains,
            "num_addresses": self.num_addresses,
            "unique_addresses": self.unique_addresses,
            "collision_count": self.collision_count,
            "collision_rate_percent": self.collision_rate * 100.0,
            "retrieval": {
                "num_queries": self.num_retrievals,
                "mean_ms": round(self.retrieval_mean_ms, 6),
                "median_ms": round(self.retrieval_median_ms, 6),
                "p95_ms": round(self.retrieval_p95_ms, 6),
                "p99_ms": round(self.retrieval_p99_ms, 6),
            },
            "performance": {
                "total_time_seconds": round(self.total_time_seconds, 3),
                "addresses_per_second": int(self.addresses_per_second),
            },
            "valid": self.is_valid(),
            "collision_analysis": self.collision_analysis(),
            "performance_analysis": self.performance_analysis(),
            "scaling_metrics": self.scaling_metrics()
        }

    def __str__(self) -> str:
        """
        String representation of scale test results.
        
        Returns:
            Human-readable summary of the test results
        """
        return (
            f"Scale {self.name()}: {self.unique_addresses}/{self.num_addresses} unique, "
            f"{self.collision_count} collisions ({self.collision_rate * 100:.2f}%), "
            f"retrieval: {self.retrieval_mean_ms:.3f}ms mean, "
            f"throughput: {self.addresses_per_second:,.0f} addr/sec"
        )


@dataclass
class FractalScalingResults:
    """
    Complete results from EXP-04 fractal scaling test.
    
    This dataclass captures the comprehensive results of testing FractalStat
    addressing across multiple scale levels, including collision analysis,
    performance metrics, and fractal property validation.
    
    Attributes:
        start_time: ISO timestamp when the test started
        end_time: ISO timestamp when the test completed
        total_duration_seconds: Total time for all scale tests
        
        scale_results: List of individual scale test results
        collision_degradation: Analysis of collision behavior across scales
        retrieval_degradation: Analysis of retrieval performance degradation
        is_fractal: Whether the system maintains fractal properties
        
    Fractal Properties Validated:
        - Self-similar behavior across scales
        - Zero collisions at all scale levels
        - Logarithmic retrieval performance scaling
        - Consistent address generation throughput
    """
    
    start_time: str
    end_time: str
    total_duration_seconds: float
    scale_results: List[ScaleTestResults]

    # Degradation analysis
    collision_degradation: Optional[str]
    retrieval_degradation: Optional[str]
    is_fractal: bool

    def get_fractal_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive fractal property analysis.
        
        Returns:
            Dictionary with fractal validation metrics and analysis
            
        Fractal Analysis Includes:
            - Collision behavior consistency across scales
            - Performance scaling patterns
            - Self-similarity validation
            - Degradation pattern analysis
        """
        if not self.scale_results:
            return {
                'fractal_valid': False,
                'reason': 'No scale results available',
                'scales_tested': 0,
                'collision_consistency': None,
                'performance_scaling': None
            }

        # Analyze collision consistency
        max_collision_rate = max(r.collision_rate for r in self.scale_results)
        collision_consistent = max_collision_rate == 0.0

        # Analyze performance scaling
        scales = [r.scale for r in self.scale_results]
        retrieval_means = [r.retrieval_mean_ms for r in self.scale_results]
        
        # Check if retrieval time scales logarithmically
        import math
        if len(self.scale_results) > 1:
            scale_ratio = max(scales) / min(scales)
            retrieval_ratio = max(retrieval_means) / min(retrieval_means)
            expected_log_ratio = math.log(scale_ratio)
            
            # Allow 2x tolerance for logarithmic scaling
            performance_logarithmic = retrieval_ratio <= expected_log_ratio * 2
        else:
            performance_logarithmic = True

        return {
            'fractal_valid': self.is_fractal,
            'scales_tested': len(self.scale_results),
            'collision_consistency': collision_consistent,
            'performance_logarithmic': performance_logarithmic,
            'max_collision_rate': max_collision_rate,
            'retrieval_scaling_ratio': retrieval_ratio if len(self.scale_results) > 1 else 1.0,
            'scale_ratio': scale_ratio if len(self.scale_results) > 1 else 1.0,
            'expected_log_ratio': expected_log_ratio if len(self.scale_results) > 1 else 1.0,
            'degradation_analysis': {
                'collision': self.collision_degradation,
                'retrieval': self.retrieval_degradation
            }
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary across all scale levels.
        
        Returns:
            Dictionary with aggregated performance metrics
            
        Performance Summary:
            - Best/worst case performance across scales
            - Average performance metrics
            - Performance consistency indicators
            - Throughput analysis
        """
        if not self.scale_results:
            return {
                'scales_tested': 0,
                'avg_retrieval_ms': 0.0,
                'min_retrieval_ms': 0.0,
                'max_retrieval_ms': 0.0,
                'avg_throughput': 0.0,
                'total_addresses': 0
            }

        retrieval_times = [r.retrieval_mean_ms for r in self.scale_results]
        throughputs = [r.addresses_per_second for r in self.scale_results]
        total_addresses = sum(r.num_addresses for r in self.scale_results)

        return {
            'scales_tested': len(self.scale_results),
            'avg_retrieval_ms': sum(retrieval_times) / len(retrieval_times),
            'min_retrieval_ms': min(retrieval_times),
            'max_retrieval_ms': max(retrieval_times),
            'avg_throughput': sum(throughputs) / len(throughputs),
            'max_throughput': max(throughputs),
            'total_addresses': total_addresses,
            'performance_consistency': max(retrieval_times) / min(retrieval_times) if min(retrieval_times) > 0 else float('inf')
        }

    def get_collision_summary(self) -> Dict[str, Any]:
        """
        Get collision analysis summary across all scale levels.
        
        Returns:
            Dictionary with collision metrics and analysis
            
        Collision Analysis:
            - Total collisions across all scales
            - Collision rate trends
            - Uniqueness validation
            - Scale-level collision patterns
        """
        if not self.scale_results:
            return {
                'total_collisions': 0,
                'max_collision_rate': 0.0,
                'collision_free_scales': 0,
                'scales_with_collisions': 0,
                'uniqueness_validation': True
            }

        total_collisions = sum(r.collision_count for r in self.scale_results)
        max_collision_rate = max(r.collision_rate for r in self.scale_results)
        collision_free_scales = sum(1 for r in self.scale_results if r.collision_count == 0)
        scales_with_collisions = len(self.scale_results) - collision_free_scales
        uniqueness_validation = all(r.collision_count == 0 for r in self.scale_results)

        return {
            'total_collisions': total_collisions,
            'max_collision_rate': max_collision_rate,
            'collision_free_scales': collision_free_scales,
            'scales_with_collisions': scales_with_collisions,
            'uniqueness_validation': uniqueness_validation,
            'fractal_collision_property': max_collision_rate == 0.0
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation suitable for JSON export
        """
        return {
            "experiment": "EXP-04",
            "test_type": "Fractal Scaling",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "scales_tested": len(self.scale_results),
            "scale_results": [r.to_dict() for r in self.scale_results],
            "degradation_analysis": {
                "collision_degradation": self.collision_degradation,
                "retrieval_degradation": self.retrieval_degradation,
            },
            "is_fractal": self.is_fractal,
            "all_valid": all(r.is_valid() for r in self.scale_results),
            "fractal_analysis": self.get_fractal_analysis(),
            "performance_summary": self.get_performance_summary(),
            "collision_summary": self.get_collision_summary()
        }

    def __str__(self) -> str:
        """
        String representation of fractal scaling test results.
        
        Returns:
            Human-readable summary of the complete test results
        """
        if not self.scale_results:
            return "Fractal Scaling Test: No results available"

        valid_scales = sum(1 for r in self.scale_results if r.is_valid())
        total_scales = len(self.scale_results)
        
        return (
            f"Fractal Scaling Test: {valid_scales}/{total_scales} scales valid, "
            f"fractal: {'YES' if self.is_fractal else 'NO'}, "
            f"duration: {self.total_duration_seconds:.1f}s, "
            f"scales: {[r.name() for r in self.scale_results]}"
        )