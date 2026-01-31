"""
EXP-14 v2: Shell-Based Atomic-Fractal Mapping

Phase 2 of fractal gravity validation: Map electron shell structure to fractal parameters.

CORRECTED DESIGN: Uses actual electron shell configuration as input, not naive Z-based mapping.

Tests whether atomic structure naturally emerges from fractal hierarchy.

Success Criteria:
- Fractal depth matches electron shell count (100% accuracy)
- Branching factor correlates with valence electrons (>0.95 correlation)
- Node count scales as branching^depth (exponential validation)
- Prediction errors decrease with shell depth (negative correlation)

MODULARIZED VERSION: This file now imports from the modularized exp14 module
for improved maintainability and reusability.
"""

import sys
from typing import List, Dict, Any, Optional

# Import from modularized structure
from fractalstat.exp14_atomic_fractal_mapping import (
    AtomicFractalMappingExperiment,
    save_results,
)

# Backward compatibility functions
def run_atomic_fractal_mapping_experiment_v2(
    elements_to_test: List[str] = None
) -> Dict[str, Any]:
    """
    Run EXP-14 v2: Shell-Based Atomic-Fractal Mapping.

    Tests whether electron shell structure naturally maps to fractal hierarchy.
    """
    experiment = AtomicFractalMappingExperiment(elements_to_test)
    return experiment.run()


def run_atomic_fractal_mapping_experiment(elements_to_test: List[str] = None) -> Dict[str, Any]:
    """Run EXP-14 v2 (shell-based mapping)."""
    return run_atomic_fractal_mapping_experiment_v2(elements_to_test)


if __name__ == "__main__":
    # Load from config or use defaults
    elements_to_test = None

    try:
        from fractalstat.config import ExperimentConfig
        config = ExperimentConfig()
        elements_to_test = config.get("EXP-14", "elements_to_test", None)
    except Exception:
        pass  # Use default (all elements)

    # Check CLI args regardless of config success (these override config)
    if "--quick" in sys.argv:
        elements_to_test = ["hydrogen", "helium", "carbon", "oxygen", "gold", "iron"]
    elif "--full" in sys.argv:
        # Use all available elements from the periodic table
        # This will be handled by the experiment class
        pass

    try:
        results = run_atomic_fractal_mapping_experiment(elements_to_test)
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("EXP-14 COMPLETE")
        print("=" * 80)

        status = "PASSED" if results["success_criteria"]["passed"] else "FAILED"
        print(f"Status: {status}")
        print(f"Output: {output_file}")

        if results["success_criteria"]["passed"]:
            print("\n🎉 SUCCESS: Electron shell structure maps perfectly to fractal hierarchy!")
            print("   This confirms that atomic structure IS fractal in nature.")
            print("   ✓ All elements have depth = shell count")
            print("   ✓ Branching correlates with valence electrons")
            print("   ✓ Node growth follows exponential pattern")
        else:
            print("\n❌ STRUCTURE MAPPING NEEDS REFINEMENT")
            print("   Some elements don't follow shell → fractal mapping.")
            print(f"   Depth accuracy: {results['structure_validation']['depth_accuracy']:.1%}")
            print(f"   Branching accuracy: {results['structure_validation']['branching_accuracy']:.1%}")

        print()

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
