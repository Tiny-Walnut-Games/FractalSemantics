"""
EXP-17: Thermodynamic Validation of Fractal Systems

Tests whether fractal simulations satisfy known thermodynamic equations.

If fractals are the fundamental structure of reality, they must obey ALL physical laws,
not just gravity. This experiment validates that fractal void/dense regions follow
thermodynamic principles.

Success Criteria:
- Fractal void regions show minimum-entropy properties
- Fractal dense regions show maximum-entropy properties
- Energy conservation (1st Law) holds in fractal interactions
- Entropy increases over time (2nd Law) in fractal evolution
- Temperature equilibration (0th Law) occurs between fractal regions

This file serves as the main entry point for EXP-17, using the modularized components
from the exp17_thermodynamic_validation module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp17_thermodynamic_validation import (
    run_thermodynamic_validation_experiment,
    save_results,
)

if __name__ == "__main__":
    try:
        results = run_thermodynamic_validation_experiment()
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("EXP-17 COMPLETE")
        print("=" * 80)

        success_rate = results["summary"]["success_rate"]
        overall_success = results["summary"]["overall_success"]

        status = "PASSED" if overall_success else "FAILED"
        print(f"Status: {status}")
        print(f"Thermodynamic validations passed: {results['summary']['validations_passed']}/{results['summary']['total_validations']}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Output: {output_file}")

        if overall_success:
            print("\n🎉 SUCCESS: Fractal systems satisfy thermodynamic laws!")
            print("   This completes the unification of physics under fractal theory.")
            print("   ✓ Energy conservation (1st Law)")
            print("   ✓ Entropy increase (2nd Law)")
            print("   ✓ Temperature equilibration (0th Law)")
            print("   ✓ Void/dense regions follow thermodynamic principles")
        else:
            print("\n❌ THERMODYNAMIC INCONSISTENCY DETECTED")
            print("   Fractal systems don't fully satisfy thermodynamic laws.")
            print("   May indicate limitations of the current fractal model.")

        print()

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
