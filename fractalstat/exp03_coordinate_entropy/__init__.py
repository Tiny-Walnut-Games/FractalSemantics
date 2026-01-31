"""
EXP-03: Coordinate Space Entropy Test - Modular Implementation

This module provides a modular implementation of the coordinate entropy test
that quantifies the information-theoretic entropy contribution of each FractalStat
dimension to the coordinate space, measuring semantic disambiguation power.

Core Hypothesis:
Each dimension contributes measurable entropy to the coordinate space. Removing or
omitting a dimension reduces entropy and semantic clarity, even if collisions remain
at 0%. Dimensions with higher entropy contribution are critical for disambiguation.

Methodology:
1. Baseline: Generate N bit-chains with all 8 dimensions, measure coordinate-level entropy (pre-hash)
2. Ablation: Remove each dimension one at a time, recompute addresses, measure entropy loss
3. Analysis: Compare entropy scores and semantic disambiguation power with vs. without each dimension
4. Validation: All dimensions should show measurable entropy contribution; some may contribute disproportionately

Usage:
    from fractalstat.exp03_coordinate_entropy import EXP03_CoordinateEntropy
    
    experiment = EXP03_CoordinateEntropy(sample_size=1000000, random_seed=42)
    results, success = experiment.run()
"""

from .entities import EXP03_Result
from .experiment import EXP03_CoordinateEntropy

__all__ = [
    'EXP03_Result',
    'EXP03_CoordinateEntropy'
]