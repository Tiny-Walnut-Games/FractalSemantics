"""
EXP-02: Retrieval Efficiency Test - Modular Implementation

This module provides a modular implementation of the retrieval efficiency test
that validates FractalStat address-based retrieval performance at scale.

Core Hypothesis:
Retrieval latency scales logarithmically or better with dataset size.

Methodology:
1. Build indexed set of N bit-chains at different scales (1M, 100M, 10B, 1T)
2. Query M random addresses (default: 1,000,000 queries)
3. Measure latency percentiles (mean, median, P95, P99)
4. Verify retrieval meets performance targets at each scale

Usage:
    from fractalstat.exp02_retrieval_efficiency import EXP02_RetrievalEfficiency
    
    experiment = EXP02_RetrievalEfficiency(query_count=1000000)
    results, success = experiment.run()
"""

from .entities import EXP02_Result
from .experiment import EXP02_RetrievalEfficiency

__all__ = [
    'EXP02_Result',
    'EXP02_RetrievalEfficiency'
]