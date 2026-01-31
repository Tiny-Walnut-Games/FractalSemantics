"""
EXP-15: Topological Conservation Laws (Modular Version)

Tests whether fractal systems conserve topology (hierarchical structure, connectivity,
branching patterns) rather than classical energy and momentum.

This is the modular version of EXP-15 that imports from the dedicated module.

CORE HYPOTHESIS:
In fractal physics, topology is the conserved quantity, not energy.
Classical Newtonian mechanics conserves energy but not topology.
Fractal mechanics conserves topology but not energy.

PHASES:
1. Define topological invariants (node count, depth, connectivity, branching)
2. Run orbital dynamics simulation and check conservation over time
3. Compare against classical Newtonian conservation laws
4. Prove topology conserved while energy is not

SUCCESS CRITERIA:
- Topology conserved over 1-year orbit (100% stability)
- Classical energy shows drift (non-conservation)
- Node count, depth, connectivity remain invariant
- Address collisions remain zero
- Structure entropy stays constant
"""

import sys
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Import from the modularized EXP-15
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp15_topological_conservation import (
    TopologicalConservationExperiment,
    save_results,
)

# Import configuration system
try:
    from fractalstat.config import ExperimentConfig
except ImportError:
    ExperimentConfig = None


def run_exp15_topological_conservation_experiment(
    systems_to_test: List[str] = None,
    approaches_to_test: List[str] = None
) -> Dict[str, any]:
    """
    Run EXP-15: Complete topological conservation experiment.

    Args:
        systems_to_test: Which systems to test
        approaches_to_test: Which approaches to test

    Returns:
        Complete experiment results
    """
    if systems_to_test is None:
        systems_to_test = ["Earth-Sun"]

    if approaches_to_test is None:
        approaches_to_test = ["Branching Vector (Ratio)"]

    start_time = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 80)
    print("EXP-15: TOPOLOGICAL CONSERVATION LAWS")
    print("=" * 80)
    print(f"Systems to test: {', '.join(systems_to_test)}")
    print(f"Approaches to test: {', '.join(approaches_to_test)}")
    print()

    # Create and run experiment
    experiment = TopologicalConservationExperiment(systems_to_test, approaches_to_test)
    results = experiment.run()

    print("\n" + "=" * 70)
    print("CROSS-ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Topology conservation confirmed: {'YES' if results['analysis']['topology_conservation_confirmed'] else 'NO'}")
    print(f"Classical energy non-conservation confirmed: {'YES' if results['analysis']['classical_energy_nonconservation_confirmed'] else 'NO'}")
    print(f"Fractal physics validated: {'YES' if results['analysis']['fractal_physics_validated'] else 'NO'}")
    print()

    return results


def save_results(results: Dict[str, any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file using the modular function."""
    from exp15_topological_conservation import save_results as modular_save_results
    return modular_save_results(results, output_file)


if __name__ == "__main__":
    # Load from config or use defaults
    systems_to_test = None
    approaches_to_test = None

    try:
        if ExperimentConfig:
            config = ExperimentConfig()
            systems_to_test = config.get("EXP-15", "systems_to_test", ["Earth-Sun"])
            approaches_to_test = config.get("EXP-15", "approaches_to_test", ["Branching Vector (Ratio)"])
    except Exception:
        pass  # Use defaults

    # Check CLI args for quick/full modes
    if "--quick" in sys.argv:
        systems_to_test = ["Earth-Sun"]
        approaches_to_test = ["Branching Vector (Ratio)"]
    elif "--full" in sys.argv:
        # Use all available systems and approaches (if configured)
        pass

    try:
        results = run_exp15_topological_conservation_experiment(
            systems_to_test=systems_to_test,
            approaches_to_test=approaches_to_test
        )
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("EXP-15 COMPLETE")
        print("=" * 80)

        status = "PASSED" if results["analysis"]["fractal_physics_validated"] else "FAILED"
        print(f"Status: {status}")
        print(f"Output: {output_file}")
        print()

        if results["analysis"]["fractal_physics_validated"]:
            print("FUNDAMENTAL BREAKTHROUGH:")
            print("✓ Topology is conserved in fractal systems")
            print("✓ Classical energy is not conserved in fractal systems")
            print("✓ Fractal physics conserves different quantities than Newtonian physics")
            print("✓ This explains why EXP-17 showed energy non-conservation")
            print()
            print("Fractal physics is validated as a fundamentally different ontology!")
        else:
            print("Topological conservation not fully demonstrated.")
            print("Further investigation needed.")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
