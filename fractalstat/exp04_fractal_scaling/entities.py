"""
EXP-04: FractalStat Fractal Scaling Test - Entities and Data Models

This module defines the data structures used in the fractal scaling test,
including configuration classes, result containers, and validation logic.
"""

import json
import time
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Reuse canonical serialization from Phase 1
from fractalstat.fractalstat_experiments import (
    BitChain,
    generate_random_bitchain,
)

secure_random = secrets.SystemRandom()


@dataclass
class ScaleTestConfig:
    """Configuration for a single scale level test."""

    scale: int  # Number of bit-chains (1K, 10K, 100K, 1M, 10M)
    num_retrievals: int  # Number of random retrieval queries
    timeout_seconds: int  # Kill test if it takes too long

    def name(self) -> str:
        """Human-readable scale name."""
        if self.scale >= 10_000_000:
            return f"{self.scale // 10_000_000}M"
        elif self.scale >= 1_000:
            return f"{self.scale // 1_000}K"
        return str(self.scale)


@dataclass
class ScaleTestResults:
    """Results from testing a single scale level."""

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
        """Check if results meet success criteria."""
        return (
            self.collision_count == 0
            and self.collision_rate == 0.0
            and self.retrieval_mean_ms < 2.0  # Sub-millisecond target
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
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
        }


@dataclass
class FractalScalingResults:
    """Complete results from EXP-04 fractal scaling test."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    scale_results: List[ScaleTestResults]

    # Degradation analysis
    collision_degradation: Optional[str]
    retrieval_degradation: Optional[str]
    is_fractal: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
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
        }