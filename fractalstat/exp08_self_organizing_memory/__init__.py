"""
EXP-08: Self-Organizing Memory Networks

This module implements self-organizing memory structures based on FractalStat coordinates.
It demonstrates emergent properties and real-world applicability through organic memory
organization without external dependencies.

Key Features:
- Self-organizing memory clusters based on semantic similarity
- Organic memory growth and clustering
- Self-organizing retrieval patterns and semantic neighborhoods
- Memory consolidation and forgetting mechanisms
- Emergent properties validation at scale

Main Classes:
- MemoryCluster: Self-organizing memory cluster based on FractalStat coordinates
- MemoryNode: Individual memory node in the self-organizing network
- ForgettingEvent: Represents a memory forgetting event
- SelfOrganizingMemoryResults: Results from the self-organizing memory test
- SelfOrganizingMemoryNetwork: Self-organizing memory network implementation
- SelfOrganizingMemoryExperiment: Main experiment runner

Usage:
    from fractalstat.exp08_self_organizing_memory import SelfOrganizingMemoryExperiment
    
    experiment = SelfOrganizingMemoryExperiment(num_memories=1000)
    results = experiment.run()
    print(f"Experiment status: {results.status}")
"""

from .entities import (
    MemoryCluster,
    MemoryNode, 
    ForgettingEvent,
    SelfOrganizingMemoryResults
)

from .experiment import (
    SelfOrganizingMemoryNetwork,
    SelfOrganizingMemoryExperiment
)

__all__ = [
    # Data structures
    'MemoryCluster',
    'MemoryNode',
    'ForgettingEvent', 
    'SelfOrganizingMemoryResults',
    
    # Core classes
    'SelfOrganizingMemoryNetwork',
    'SelfOrganizingMemoryExperiment',
]

__version__ = "1.0.0"
__author__ = "FractalStat Team"