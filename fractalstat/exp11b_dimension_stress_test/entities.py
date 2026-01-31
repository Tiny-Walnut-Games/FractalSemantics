"""
EXP-11b: Dimensional Collision Stress Test - Data Entities

This module contains all the data structures and entities used in the dimensional
collision stress test experiment. These entities represent the core components
of the stress testing system including test results, dimension configurations,
and experiment results.

Classes:
- StressTestResult: Results for a single stress test configuration
- DimensionStressTestResult: Complete results from dimensional collision stress testing
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

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Dimension constants
FRACTALSTAT_DIMENSIONS = [
    "realm", "lineage", "adjacency", "horizon", "luminosity",
    "polarity", "dimensionality", "alignment"
]

# Test configuration constants
DEFAULT_SAMPLE_SIZE = 10_000
MAX_DIVERSITY_SAMPLE_SIZE = 1_000

# Coordinate range limits for stress testing
COORDINATE_RANGE_LIMITS = {
    "limited": 0.1,  # ±10%
    "very_limited": 0.01,  # ±1%
}

# Common dimension subsets for testing
DIMENSION_SUBSETS = {
    "minimal_3d": ["realm", "lineage", "horizon"],
    "minimal_2d": ["realm", "lineage"],
    "single_dimension": ["realm"],
    "continuous_only": ["luminosity", "dimensionality", "adjacency"],
    "categorical_only": ["realm", "horizon"],
    "all_dimensions": FRACTALSTAT_DIMENSIONS
}

# Test scenario configurations
@dataclass
class TestScenario:
    """Configuration for a single test scenario."""
    name: str
    description: str
    use_unique_id: bool
    use_unique_state: bool
    coordinate_range_limit: Optional[float]
    dimensions: List[str]


@dataclass
class StressTestResult:
    """Results for a single stress test configuration."""

    test_name: str
    dimension_count: int
    dimensions_used: List[str]
    sample_size: int
    unique_addresses: int
    collisions: int
    collision_rate: float
    max_collisions_per_address: int
    coordinate_diversity: float  # 0.0 to 1.0, how varied the coordinates are
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionStressTestResult:
    """Complete results from EXP-11b dimension stress testing."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    test_results: List[StressTestResult]
    key_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": "EXP-11b",
            "test_type": "Dimensional Collision Stress Test",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "key_findings": self.key_findings,
            "test_results": [r.to_dict() for r in self.test_results],
        }