"""
EXP-09: FractalStat Performance Under Memory Pressure - Data Entities

This module contains all the data structures and entities used in the memory pressure
testing experiment. These entities represent the core components of the memory
pressure testing system including metrics, test phases, optimization strategies,
and experiment results.

Classes:
- MemoryPressureMetrics: Metrics collected during memory pressure testing
- StressTestPhase: Represents a phase in the memory stress testing
- MemoryOptimization: Memory optimization strategy applied during testing
- MemoryPressureResults: Results from the memory pressure test
"""

import json
import time
import secrets
import gc
import psutil
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import statistics
import threading

from fractalstat.fractalstat_entity import generate_random_bitchain, BitChain

secure_random = secrets.SystemRandom()


@dataclass
class MemoryPressureMetrics:
    """Metrics collected during memory pressure testing."""
    
    timestamp: float
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    active_objects: int
    garbage_collections: int
    retrieval_latency_ms: float
    storage_efficiency: float
    fragmentation_ratio: float


@dataclass
class StressTestPhase:
    """Represents a phase in the memory stress testing."""
    
    phase_name: str
    target_memory_mb: int
    duration_seconds: int
    load_pattern: str  # "linear", "exponential", "spike"
    optimization_enabled: bool
    expected_behavior: str


@dataclass
class MemoryOptimization:
    """Memory optimization strategy applied during testing."""
    
    strategy_name: str
    description: str
    memory_reduction_target: float  # Target reduction percentage
    performance_impact: str  # "minimal", "moderate", "significant"
    enabled: bool = True


@dataclass
class MemoryPressureResults:
    """Results from EXP-09 memory pressure test."""
    
    experiment: str = "EXP-09"
    title: str = "FractalStat Performance Under Memory Pressure"
    timestamp: str = ""
    status: str = "PASS"
    
    # Test configuration
    total_duration_seconds: float = 0.0
    max_memory_target_mb: int = 0
    optimization_strategies: List[str] = field(default_factory=list)
    
    # Performance metrics
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    stress_performance: Dict[str, float] = field(default_factory=dict)
    degradation_ratio: float = 0.0
    recovery_time_seconds: float = 0.0
    
    # Memory management
    peak_memory_usage_mb: float = 0.0
    memory_efficiency_score: float = 0.0
    garbage_collection_effectiveness: float = 0.0
    fragmentation_score: float = 0.0
    
    # System resilience
    stability_score: float = 0.0
    breaking_point_memory_mb: Optional[float] = None
    graceful_degradation: bool = False
    optimization_improvement: float = 0.0
    
    # Detailed metrics
    pressure_phases: List[Dict[str, Any]] = field(default_factory=list)
    optimization_results: List[Dict[str, Any]] = field(default_factory=list)
    memory_timeline: List[MemoryPressureMetrics] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert memory timeline to list of dicts
        result['memory_timeline'] = [m.__dict__ for m in self.memory_timeline]
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)