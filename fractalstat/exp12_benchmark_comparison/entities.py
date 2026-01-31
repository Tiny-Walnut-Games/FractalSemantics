"""
EXP-12: Benchmark Comparison - Data Entities

This module contains all the data structures and entities used in the benchmark
comparison experiment. These entities represent the core components of the
benchmarking system including test results, system configurations, and
comparative analysis results.

Classes:
- SystemBenchmarkResult: Results for a single system benchmark
- BenchmarkComparisonResult: Complete results from benchmark comparison
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
# BENCHMARK SYSTEM BASE CLASSES
# ============================================================================


class BenchmarkSystem:
    """Base class for benchmark systems."""

    def __init__(self, name: str):
        self.name = name
        self.storage: Dict[str, Any] = {}

    def generate_address(self, entity: Any) -> str:
        """Generate address/identifier for entity."""
        raise NotImplementedError

    def store(self, address: str, entity: Any) -> None:
        """Store entity at address."""
        self.storage[address] = entity

    def retrieve(self, address: str) -> Optional[Any]:
        """Retrieve entity by address."""
        return self.storage.get(address)

    def get_storage_size(self, entity: Any) -> int:
        """Get storage size in bytes for entity."""
        return len(json.dumps({"address": "placeholder", "data": str(entity)}))

    def get_semantic_expressiveness(self) -> float:
        """Get semantic expressiveness score (0.0 to 1.0)."""
        return 0.0

    def get_relationship_support(self) -> float:
        """Get relationship support score (0.0 to 1.0)."""
        return 0.0

    def get_query_flexibility(self) -> float:
        """Get query flexibility score (0.0 to 1.0)."""
        return 0.0


class UUIDSystem(BenchmarkSystem):
    """UUID/GUID system (128-bit random identifiers)."""

    def __init__(self):
        super().__init__("UUID")

    def generate_address(self, entity: Any) -> str:
        """Generate random UUID."""
        import uuid
        return str(uuid.uuid4())

    def get_semantic_expressiveness(self) -> float:
        return 0.0  # No semantic meaning

    def get_relationship_support(self) -> float:
        return 0.0  # No built-in relationships

    def get_query_flexibility(self) -> float:
        return 0.1  # Only exact match queries


class SHA256System(BenchmarkSystem):
    """SHA-256 content-addressable storage (Git-style)."""

    def __init__(self):
        super().__init__("SHA256")

    def generate_address(self, entity: Any) -> str:
        """Generate SHA-256 hash of entity content."""
        import hashlib
        from fractalstat.fractalstat_entity import canonical_serialize
        
        if isinstance(entity, BitChain):
            content = canonical_serialize(entity.to_canonical_dict())
        else:
            content = json.dumps(entity, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_semantic_expressiveness(self) -> float:
        return 0.1  # Content-based, but no semantic structure

    def get_relationship_support(self) -> float:
        return 0.0  # No built-in relationships

    def get_query_flexibility(self) -> float:
        return 0.2  # Exact match + content verification


class VectorDBSystem(BenchmarkSystem):
    """Vector database system (similarity search)."""

    def __init__(self):
        super().__init__("VectorDB")
        self.embeddings: Dict[str, List[float]] = {}

    def generate_address(self, entity: Any) -> str:
        """Generate UUID + store embedding."""
        import uuid
        addr = str(uuid.uuid4())
        # Simulate embedding (use entity properties as vector)
        if isinstance(entity, BitChain):
            embedding = [
                entity.coordinates.lineage,
                entity.coordinates.luminosity,
                entity.coordinates.dimensionality,
            ]
            self.embeddings[addr] = embedding
        return addr

    def get_semantic_expressiveness(self) -> float:
        return 0.7  # Good semantic similarity via embeddings

    def get_relationship_support(self) -> float:
        return 0.3  # Limited relationship support

    def get_query_flexibility(self) -> float:
        return 0.6  # Similarity search + filtering


class GraphDBSystem(BenchmarkSystem):
    """Graph database system (relationship traversal)."""

    def __init__(self):
        super().__init__("GraphDB")
        self.edges: Dict[str, List[str]] = {}

    def generate_address(self, entity: Any) -> str:
        """Generate UUID + store relationships."""
        import uuid
        addr = str(uuid.uuid4())
        # Simulate edges (use adjacency if available)
        if isinstance(entity, BitChain):
            self.edges[addr] = entity.coordinates.adjacency.copy()
        return addr

    def get_semantic_expressiveness(self) -> float:
        return 0.4  # Some semantic structure via relationships

    def get_relationship_support(self) -> float:
        return 0.9  # Excellent relationship support

    def get_query_flexibility(self) -> float:
        return 0.7  # Graph traversal + pattern matching


class RDBMSSystem(BenchmarkSystem):
    """Traditional RDBMS with indexes."""

    def __init__(self):
        super().__init__("RDBMS")
        self.indexes: Dict[str, List[str]] = {}

    def generate_address(self, entity: Any) -> str:
        """Generate auto-increment ID + build indexes."""
        import uuid
        addr = str(uuid.uuid4())  # Simulate auto-increment
        # Simulate indexes on key fields
        if isinstance(entity, BitChain):
            realm_key = f"realm:{entity.coordinates.realm}"
            if realm_key not in self.indexes:
                self.indexes[realm_key] = []
            self.indexes[realm_key].append(addr)
        return addr

    def get_semantic_expressiveness(self) -> float:
        return 0.5  # Structured schema with typed fields

    def get_relationship_support(self) -> float:
        return 0.6  # Foreign keys + joins

    def get_query_flexibility(self) -> float:
        return 0.8  # SQL queries with complex predicates


class FractalStatSystem(BenchmarkSystem):
    """FractalStat 7-dimensional addressing system."""

    def __init__(self):
        super().__init__("FractalStat")

    def generate_address(self, entity: Any) -> str:
        """Generate FractalStat address."""
        from fractalstat.fractalstat_entity import compute_address_hash
        
        if isinstance(entity, BitChain):
            return entity.compute_address()
        else:
            # Fallback for non-BitChain entities
            return compute_address_hash({"data": str(entity)})

    def get_semantic_expressiveness(self) -> float:
        return 0.95  # Excellent: 7 semantic dimensions

    def get_relationship_support(self) -> float:
        return 0.8  # Good: adjacency dimension + coordinate proximity

    def get_query_flexibility(self) -> float:
        return 0.9  # Excellent: multi-dimensional queries


# ============================================================================
# BENCHMARK DATA STRUCTURES
# ============================================================================


@dataclass
class SystemBenchmarkResult:
    """Results for a single system benchmark."""

    system_name: str
    scale: int
    num_queries: int

    # Uniqueness metrics
    unique_addresses: int
    collisions: int
    collision_rate: float

    # Retrieval metrics
    mean_retrieval_latency_ms: float
    median_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    p99_retrieval_latency_ms: float

    # Storage metrics
    avg_storage_bytes_per_entity: int
    total_storage_bytes: int

    # Semantic capabilities (0.0 to 1.0)
    semantic_expressiveness: float
    relationship_support: float
    query_flexibility: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)


@dataclass
class BenchmarkComparisonResult:
    """Complete results from EXP-12 benchmark comparison."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    sample_size: int
    scales_tested: List[int]
    num_queries: int
    systems_tested: List[str]

    # Per-system results
    system_results: List[SystemBenchmarkResult]

    # Comparative analysis
    best_collision_rate_system: str
    best_retrieval_latency_system: str
    best_storage_efficiency_system: str
    best_semantic_expressiveness_system: str
    best_overall_system: str

    # FractalStat positioning
    fractalstat_rank_collision: int  # 1 = best
    fractalstat_rank_retrieval: int
    fractalstat_rank_storage: int
    fractalstat_rank_semantic: int
    fractalstat_overall_score: float  # 0.0 to 1.0

    # Key findings
    major_findings: List[str] = field(default_factory=list)
    fractalstat_competitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "experiment": "EXP-12",
            "test_type": "Benchmark Comparison",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "sample_size": self.sample_size,
            "scales_tested": self.scales_tested,
            "num_queries": self.num_queries,
            "systems_tested": self.systems_tested,
            "comparative_analysis": {
                "best_collision_rate": self.best_collision_rate_system,
                "best_retrieval_latency": self.best_retrieval_latency_system,
                "best_storage_efficiency": self.best_storage_efficiency_system,
                "best_semantic_expressiveness": self.best_semantic_expressiveness_system,
                "best_overall": self.best_overall_system,
            },
            "fractalstat_positioning": {
                "rank_collision": self.fractalstat_rank_collision,
                "rank_retrieval": self.fractalstat_rank_retrieval,
                "rank_storage": self.fractalstat_rank_storage,
                "rank_semantic": self.fractalstat_rank_semantic,
                "overall_score": round(self.fractalstat_overall_score, 3),
                "competitive": self.fractalstat_competitive,
            },
            "major_findings": self.major_findings,
            "system_results": [r.to_dict() for r in self.system_results],
        }