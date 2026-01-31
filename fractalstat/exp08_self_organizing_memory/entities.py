"""
EXP-08: Self-Organizing Memory Networks - Data Entities

This module contains all the data structures and entities used in the self-organizing
memory network experiment. These entities represent the core components of the
memory system including clusters, nodes, forgetting events, and experiment results.

Classes:
- MemoryCluster: Self-organizing memory cluster based on FractalStat coordinates
- MemoryNode: Individual memory node in the self-organizing network  
- ForgettingEvent: Represents a memory forgetting event
- SelfOrganizingMemoryResults: Results from the self-organizing memory test
"""

import json
import time
import secrets
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import statistics

import sys
import os

# Add the current directory to Python path to allow direct imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractalstat.fractalstat_entity import generate_random_bitchain, BitChain


secure_random = secrets.SystemRandom()


@dataclass
class MemoryCluster:
    """A self-organizing memory cluster based on FractalStat coordinates."""
    
    cluster_id: str
    representative_address: str
    member_addresses: List[str] = field(default_factory=list)
    semantic_cohesion: float = 0.0  # 0.0 to 1.0
    activity_level: float = 0.0     # Memory usage frequency
    last_accessed: float = 0.0      # Timestamp
    consolidation_level: float = 0.0  # 0.0 (raw) to 1.0 (highly consolidated)
    
    def add_member(self, address: str):
        """Add a member to this cluster."""
        if address not in self.member_addresses:
            self.member_addresses.append(address)
    
    def update_activity(self, timestamp: float):
        """Update cluster activity."""
        self.last_accessed = timestamp
        self.activity_level = min(1.0, self.activity_level + 0.1)
    
    def consolidate(self):
        """Apply memory consolidation to reduce overhead."""
        if len(self.member_addresses) > 10:
            # Keep only most active members
            self.consolidation_level = min(1.0, self.consolidation_level + 0.1)
            # Sort by activity and keep top 70%
            cutoff = max(5, int(len(self.member_addresses) * 0.7))
            self.member_addresses = self.member_addresses[:cutoff]


@dataclass
class MemoryNode:
    """Individual memory node in the self-organizing network."""
    
    address: str
    content: Dict[str, Any]
    coordinates: Dict[str, Any]
    activation_count: int = 0
    last_accessed: float = 0.0
    semantic_neighbors: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None


@dataclass
class ForgettingEvent:
    """Represents a memory forgetting event."""
    
    address: str
    reason: str  # "decay", "consolidation", "overload"
    timestamp: float
    memory_value: float  # Value before forgetting


@dataclass
class SelfOrganizingMemoryResults:
    """Results from EXP-08 self-organizing memory test."""
    
    experiment: str = "EXP-08"
    title: str = "Self-Organizing Memory Networks"
    timestamp: str = ""
    status: str = "PASS"
    
    # Memory organization metrics
    total_memories: int = 0
    num_clusters: int = 0
    avg_cluster_size: float = 0.0
    semantic_cohesion_score: float = 0.0
    cluster_efficiency: float = 0.0
    
    # Retrieval performance
    retrieval_efficiency: float = 0.0
    semantic_retrieval_accuracy: float = 0.0
    self_organization_improvement: float = 0.0
    
    # Memory management
    consolidation_ratio: float = 0.0
    forgetting_events: int = 0
    memory_pressure: float = 0.0
    storage_overhead_reduction: float = 0.0
    
    # Emergent properties
    emergent_intelligence_score: float = 0.0
    organic_growth_validated: bool = False
    network_connectivity: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)