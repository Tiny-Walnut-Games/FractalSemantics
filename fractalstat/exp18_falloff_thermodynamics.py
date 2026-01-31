"""
EXP-18: Falloff Injection in Thermodynamics

Tests whether applying the same falloff formula used in gravity to thermodynamic
measurements makes fractal thermodynamics behave more like classical thermodynamics.

If gravity and thermodynamics both emerge from fractal structure, then injecting
the same falloff should make thermodynamic behavior more "classical" (energy conserved,
entropy increasing, temperatures equilibrating).

Success Criteria:
- With falloff injection, energy conservation improves
- With falloff injection, entropy shows classical increase
- With falloff injection, temperature equilibration occurs
- With falloff injection, void/dense entropy follows classical expectations

This file serves as the main entry point for EXP-18, using the modularized components
from the exp18_falloff_thermodynamics module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp18_falloff_thermodynamics import (
    run_falloff_thermodynamics_experiment,
    save_results,
)

if __name__ == "__main__":
    try:
        # Test with the same falloff exponent used in gravity (2.0)
        results = run_falloff_thermodynamics_experiment(falloff_exponent=2.0)
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("EXP-18 COMPLETE")
        print("=" * 80)

        comparison = results["comparison"]
        success = results["success_criteria"]["passed"]

        status = "PASSED" if success else "FAILED"
        print(f"Status: {status}")
        print(f"Falloff injection improves thermodynamics: {comparison['falloff_improves_thermodynamics']}")
        print(f"Validations without falloff: {comparison['passed_no_falloff']}/4")
        print(f"Validations with falloff: {comparison['passed_with_falloff']}/4")
        print(f"Improvement: {comparison['improvement']} validations")
        print(f"Output: {output_file}")

        if success:
            print("\n🎉 SUCCESS: Falloff injection improves thermodynamic behavior!")
            print("   This confirms that gravity and thermodynamics share the same falloff mechanism.")
            print("   ✓ Same falloff formula works for both energy and gravitational interactions")
        else:
            print("\n❌ NO IMPROVEMENT: Falloff injection doesn't help thermodynamics")
            print("   Gravity and thermodynamics may have different falloff characteristics.")

        print()

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
