"""
EXP-06: Entanglement Detection - Modular Implementation

This module provides a modular implementation of the entanglement detection experiment
that implements the mathematical framework for detecting semantic entanglement between
bit-chains. The experiment validates the core hypothesis that FractalStat coordinates
can be used to detect meaningful relationships between entities through mathematical
similarity measures.

Core Mathematical Framework:
E(B1, B2) = 0.5·P + 0.15·R + 0.2·A + 0.1·L + 0.05·ell

Where:
  P = Polarity Resonance (cosine similarity)
  R = Realm Affinity (categorical adjacency)
  A = Adjacency Overlap (Jaccard similarity)
  L = Luminosity Proximity (density distance)
  ell = Lineage Affinity (exponential decay)

Key Validation Points:
- Polarity resonance as primary entanglement signal (weight: 0.5)
- Realm affinity for cross-realm compatibility (weight: 0.15)
- Adjacency overlap for "guilt by association" (weight: 0.2)
- Luminosity proximity for compression state matching (weight: 0.1)
- Lineage affinity for generational relationships (weight: 0.05)

Success Criteria:
- Precision ≥ 70% for entanglement detection
- Recall ≥ 60% for relationship identification
- F1 score ≥ 65% for balanced performance
- Runtime efficiency for large-scale analysis

Usage:
    from fractalstat.exp06_entanglement_detection import EntanglementDetector, compute_entanglement_score
    
    detector = EntanglementDetector(threshold=0.85)
    entangled_pairs = detector.detect(bitchains)
    print(f"Found {len(entangled_pairs)} entangled pairs")
"""

from .entities import (
    EntanglementScore,
    ValidationResult,
    REALM_ADJACENCY
)
from .experiment import (
    EntanglementDetector,
    compute_entanglement_score,
    run_experiment,
    save_results,
    main
)

__all__ = [
    'EntanglementScore',
    'ValidationResult',
    'REALM_ADJACENCY',
    'EntanglementDetector',
    'compute_entanglement_score',
    'run_experiment',
    'save_results',
    'main'
]