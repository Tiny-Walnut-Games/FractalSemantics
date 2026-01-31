"""
EXP-07: LUCA Bootstrap Test - Modular Implementation

This module provides a modular implementation of the LUCA bootstrap test that
proves the system can be reconstructed from LUCA (Last Universal Common Ancestor).
The experiment validates that FractalStat can compress a full system to an
irreducible minimum and then bootstrap back to the complete system without
information loss.

Core Hypothesis:
The FractalStat system is self-contained and fractal, allowing complete
reconstruction from a minimal LUCA state while preserving all critical
information and maintaining system integrity.

Key Validation Points:
- Compression to LUCA state preserves essential information
- Bootstrap reconstruction achieves 100% entity recovery
- Lineage continuity maintained through compression/expansion cycles
- Fractal properties preserved across scales
- System demonstrates self-similarity and scale invariance

Success Criteria:
- Entity recovery rate ≥ 100% (perfect reconstruction)
- Lineage recovery rate ≥ 100% (continuity preserved)
- Realm recovery rate ≥ 100% (structural integrity)
- Dimensionality recovery rate ≥ 100% (fractal depth preserved)
- Multiple bootstrap cycles without degradation
- Compression ratio > 0 (meaningful compression achieved)

Usage:
    from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester, TestBitChain
    
    tester = LUCABootstrapTester()
    results = tester.run_comprehensive_test()
    print(f"Test passed: {results.status == 'PASS'}")
"""

from .entities import (
    TestBitChain,
    LUCABootstrapResult
)
from .experiment import (
    LUCABootstrapTester,
    save_results,
    main
)

__all__ = [
    'TestBitChain',
    'LUCABootstrapResult',
    'LUCABootstrapTester',
    'save_results',
    'main'
]