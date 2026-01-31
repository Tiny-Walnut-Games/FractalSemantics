"""
EXP-11: Dimension Cardinality Analysis - Data Entities

This module contains all the data structures and entities used in the dimension
cardinality analysis experiment. These entities represent the core components
of the dimension testing system including test results, dimension configurations,
and experiment results.

Classes:
- DimensionTestResult: Results for a single dimension count test
- DimensionCardinalityResult: Complete results from dimension cardinality analysis
"""

import json
import time
import secrets
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import statistics

from fractalstat.fractalstat_entity import generate_random_bitchain, BitChain

secure_random = secrets.SystemRandom()


@dataclass
class DimensionTestResult:
    """Results for a single dimension count test."""
    
    dimension_count: int
    dimensions_used: List[str]
    sample_size: int
    unique_addresses: int
    collisions: int
    collision_rate: float
    mean_retrieval_latency_ms: float
    median_retrieval_latency_ms: float
    avg_storage_bytes: int
    storage_overhead_per_dimension: float
    semantic_expressiveness_score: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)


@dataclass
class DimensionCardinalityResult:
    """Complete results from EXP-11 dimension cardinality analysis."""
    
    start_time: str
    end_time: str
    total_duration_seconds: float
    sample_size: int
    dimension_counts_tested: List[int]
    test_iterations: int

    # Per-dimension-count results
    dimension_results: List[DimensionTestResult]

    # Aggregate analysis
    optimal_dimension_count: int
    optimal_collision_rate: float
    optimal_retrieval_latency_ms: float
    optimal_storage_efficiency: float
    diminishing_returns_threshold: int  # Dimension count where returns diminish

    # Key findings
    major_findings: List[str] = field(default_factory=list)
    seven_dimensions_justified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "experiment": "EXP-11",
            "test_type": "Dimension Cardinality Analysis",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "sample_size": self.sample_size,
            "dimension_counts_tested": self.dimension_counts_tested,
            "test_iterations": self.test_iterations,
            "optimal_analysis": {
                "optimal_dimension_count": self.optimal_dimension_count,
                "optimal_collision_rate": round(self.optimal_collision_rate, 6),
                "optimal_retrieval_latency_ms": round(
                    self.optimal_retrieval_latency_ms, 4
                ),
                "optimal_storage_efficiency": round(self.optimal_storage_efficiency, 3),
                "diminishing_returns_threshold": self.diminishing_returns_threshold,
            },
            "seven_dimensions_justified": self.seven_dimensions_justified,
            "major_findings": self.major_findings,
            "dimension_results": [r.to_dict() for r in self.dimension_results],
        }