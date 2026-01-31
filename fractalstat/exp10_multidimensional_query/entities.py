"""
EXP-10: Multi-Dimensional Query Optimization - Data Entities

This module contains all the data structures and entities used in the multi-dimensional
query optimization experiment. These entities represent the core components of the
query system including query patterns, results, optimization strategies, and experiment
results.

Classes:
- QueryPattern: Definition of a multi-dimensional query pattern
- QueryResult: Results from executing a multi-dimensional query
- QueryOptimizer: Query optimization strategy for multi-dimensional queries
- MultiDimensionalQueryResults: Results from the multi-dimensional query optimization test
"""

import json
import time
import secrets
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import statistics

from fractalstat.fractalstat_entity import generate_random_bitchain, BitChain

secure_random = secrets.SystemRandom()


@dataclass
class QueryPattern:
    """Definition of a multi-dimensional query pattern."""
    
    pattern_name: str
    description: str
    dimensions_used: List[str]
    complexity_level: str  # "simple", "medium", "complex", "expert"
    real_world_use_case: str


@dataclass
class QueryResult:
    """Results from executing a multi-dimensional query."""
    
    query_id: str
    pattern_name: str
    execution_time_ms: float
    results_count: int
    precision_score: float  # 0.0 to 1.0
    recall_score: float    # 0.0 to 1.0
    f1_score: float        # Combined precision/recall
    memory_usage_mb: float
    cpu_time_ms: float


@dataclass
class QueryOptimizer:
    """Query optimization strategy for multi-dimensional queries."""
    
    strategy_name: str
    description: str
    optimization_type: str  # "indexing", "caching", "pruning", "parallelization"
    expected_improvement: float  # Expected performance improvement
    complexity_overhead: str   # "low", "medium", "high"


@dataclass
class MultiDimensionalQueryResults:
    """Results from EXP-10 multi-dimensional query optimization test."""
    
    experiment: str = "EXP-10"
    title: str = "Multi-Dimensional Query Optimization"
    timestamp: str = ""
    status: str = "PASS"
    
    # Dataset information
    dataset_size: int = 0
    dimensions_coverage: Dict[str, int] = field(default_factory=dict)
    coordinate_diversity: float = 0.0
    
    # Query performance metrics
    avg_query_time_ms: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1_score: float = 0.0
    query_throughput_qps: float = 0.0
    
    # Optimization effectiveness
    optimization_strategies: List[str] = field(default_factory=list)
    optimization_improvement: float = 0.0
    indexing_efficiency: float = 0.0
    caching_effectiveness: float = 0.0
    
    # Real-world applicability
    use_case_validation: Dict[str, bool] = field(default_factory=dict)
    practical_value_score: float = 0.0
    scalability_score: float = 0.0
    
    # Detailed results
    query_results: List[QueryResult] = field(default_factory=list)
    optimizer_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert query results to list of dicts
        result['query_results'] = [qr.__dict__ for qr in self.query_results]
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)