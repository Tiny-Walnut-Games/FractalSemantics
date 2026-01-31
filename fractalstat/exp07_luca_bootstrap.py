#!/usr/bin/env python3
# pylint: disable=W0621,W0212,W0718,
"""
EXP-07: LUCA Bootstrap Test
Goal: Prove we can reconstruct entire system from LUCA (Last Universal Common Ancestor)

What it tests:
- Compress full system to LUCA (irreducible minimum)
- Bootstrap: Can we unfold LUCA back to full system?
- Compare: bootstrapped system == original?
- Fractal verification: same structure at different scales

Expected Result:
- Full reconstruction possible
- No information loss
- System is self-contained and fractal
- LUCA acts as stable bootstrap origin
"""

import json
import time
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Import FractalStat components
from fractalstat.exp07_luca_bootstrap import (
    LUCABootstrapTester as ModularLUCABootstrapTester,
    save_results as modular_save_results
)


# ============================================================================
# Backward Compatibility Wrapper
# ============================================================================

# For backward compatibility, we provide the same interface as the original file
# but delegate to the modular implementation

def LUCABootstrapTester():
    """Backward compatibility wrapper for LUCABootstrapTester."""
    return ModularLUCABootstrapTester()


def save_results(results, output_file=None):
    """Backward compatibility wrapper for save_results."""
    return modular_save_results(results, output_file)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run EXP-07 LUCA Bootstrap Test."""
    import sys

    tester = ModularLUCABootstrapTester()
    results = tester.run_comprehensive_test()

    # Save complete results to JSON file
    modular_save_results(results.to_dict())

    # Set exit code based on test status for orchestrator
    success = results.status == "PASS"
    exit_code = 0 if success else 1

    # Print summary with celebration at the end
    print("\n[SUMMARY]")
    print("-" * 70)
    print(json.dumps(results.results, indent=2))

    # Celebration at the end
    if success:
        print("\n[Success] EXP-07 LUCA Bootstrap: PERFECT RECONSTRUCTION ACHIEVED!")
        print(f"  - 100% All {results.results['bootstrap']['bootstrapped_count']} entities recovered")
        print("   - Lineage continuity: VERIFIED")
        print("   - Fractal properties: CONFIRMED")
        print("   - Bootstrap stability: COMPLETED")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
