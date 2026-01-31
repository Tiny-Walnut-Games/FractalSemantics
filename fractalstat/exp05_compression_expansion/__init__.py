"""
EXP-05: Bit-Chain Compression/Expansion Losslessness Validation - Modular Implementation

This module provides a modular implementation of the compression/expansion validation
experiment that tests whether FractalStat bit-chains can be compressed through the full
pipeline (fragments → clusters → glyphs → mist) and then expanded back to original
coordinates without information loss.

Core Hypothesis:
The FractalStat compression pipeline maintains lossless information preservation
through all stages, allowing complete reconstruction of original coordinates and
preservation of provenance, narrative meaning, and luminosity characteristics.

Methodology:
1. Generate random bit-chains with full FractalStat coordinates
2. Compress through pipeline stages: fragments → clusters → glyphs → mist
3. Attempt reconstruction from mist back to original coordinates
4. Validate provenance chain integrity and coordinate accuracy
5. Measure compression ratios and information preservation metrics

Success Criteria:
- Provenance chain integrity maintained through all compression stages
- FractalStat coordinate reconstruction accuracy ≥ 40%
- Narrative preservation via embeddings and affect ≥ 90%
- Compression ratio ≥ 2.0x for practical efficiency
- Luminosity decay controlled and predictable

Usage:
    from fractalstat.exp05_compression_expansion import run_compression_expansion_test
    
    results = run_compression_expansion_test(num_bitchains=1000000)
    print(f"Lossless system: {results.is_lossless}")
"""

from .entities import (
    CompressionStage,
    BitChainCompressionPath,
    CompressionExperimentResults
)
from .experiment import (
    CompressionPipeline,
    run_compression_expansion_test,
    save_results
)

__all__ = [
    'CompressionStage',
    'BitChainCompressionPath',
    'CompressionExperimentResults',
    'CompressionPipeline',
    'run_compression_expansion_test',
    'save_results'
]