"""
EXP-08: Self-Organizing Memory Networks - Modular Version

This is the modular version of the self-organizing memory networks experiment.
It imports all functionality from the modular structure while maintaining
backward compatibility with the original API.

This file serves as a compatibility layer and demonstrates how to use the
modular components.
"""

# Import all functionality from the modular structure
from fractalstat.exp08_self_organizing_memory import (
    # Data structures
    MemoryCluster,
    MemoryNode, 
    ForgettingEvent,
    SelfOrganizingMemoryResults,
    
    # Core classes
    SelfOrganizingMemoryNetwork,
    SelfOrganizingMemoryExperiment,
    
    # Utility functions
    save_results,
    main
)

# Re-export everything to maintain backward compatibility
__all__ = [
    # Data structures
    'MemoryCluster',
    'MemoryNode',
    'ForgettingEvent', 
    'SelfOrganizingMemoryResults',
    
    # Core classes
    'SelfOrganizingMemoryNetwork',
    'SelfOrganizingMemoryExperiment',
    
    # Utility functions
    'save_results',
    'main',
]

# Maintain the original module docstring for compatibility
__doc__ = """
EXP-08: Self-Organizing Memory Networks

Demonstrates FractalStat's ability to create self-organizing memory structures
without external dependencies. This experiment validates emergent properties
and real-world applicability through organic memory organization.

Hypothesis:
FractalStat coordinates enable self-organizing memory networks where:
- Memory clusters form naturally based on semantic similarity
- Retrieval patterns emerge organically without explicit indexing
- Memory consolidation and forgetting mechanisms improve efficiency
- Performance scales gracefully with organic growth patterns

Methodology:
1. Generate diverse bit-chains representing different memory entities
2. Simulate organic memory growth and clustering based on FractalStat coordinates
3. Test self-organizing retrieval patterns and semantic neighborhoods
4. Measure memory consolidation efficiency and forgetting mechanisms
5. Validate emergent properties at scale

Success Criteria:
- Memory clusters form with >80% semantic coherence
- Retrieval efficiency improves through self-organization
- Memory consolidation reduces storage overhead by >50%
- Forgetting mechanisms maintain optimal memory pressure
- Emergent properties demonstrate system intelligence
"""