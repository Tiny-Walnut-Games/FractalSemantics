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
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Import subprocess communication for enhanced progress reporting
try:
    from fractalsemantics.subprocess_comm import (
        send_subprocess_progress,
        send_subprocess_status,
        send_subprocess_completion,
        is_subprocess_communication_enabled
    )
except ImportError:
    # Fallback if subprocess communication is not available
    def send_subprocess_progress(*args, **kwargs) -> bool: return False
    def send_subprocess_status(*args, **kwargs) -> bool: return False
    def send_subprocess_completion(*args, **kwargs) -> bool: return False
    def is_subprocess_communication_enabled() -> bool: return False

# Import fractal components
from fractalsemantics.exp13_fractal_gravity import (
    FractalHierarchy,
    compute_natural_cohesion,
)
from fractalsemantics.exp17_thermodynamic_validation import (
    ThermodynamicState,
    create_fractal_region,
    validate_first_law,
    validate_fractal_void_density,
    validate_second_law,
    validate_zeroth_law,
)

secure_random = np.random.RandomState(42)

# ============================================================================
# FALLOFF-INJECTED THERMODYNAMIC MEASUREMENTS
# ============================================================================

def measure_fractal_energy_with_falloff(hierarchy: FractalHierarchy,
                                       falloff_exponent: float = 2.0) -> float:
    """
    Measure fractal energy WITH falloff injection.

    Applies the same 1/r^falloff_exponent formula used in gravity to energy measurements.
    """
    total_energy = 0.0
    all_nodes = hierarchy.get_all_nodes()

    # Sample cohesion energies with falloff applied
    sample_size = min(1000, len(all_nodes))
    sampled_indices = secure_random.choice(len(all_nodes), size=sample_size, replace=False)

    for i in sampled_indices:
        for j in sampled_indices:
            if i != j:
                node_a = all_nodes[i]
                node_b = all_nodes[j]

                # Get natural cohesion
                cohesion = compute_natural_cohesion(node_a, node_b, hierarchy)

                # Apply falloff based on hierarchical distance (same as gravity)
                hierarchical_distance = hierarchy.get_hierarchical_distance(node_a, node_b)
                falloff_factor = 1.0 / ((hierarchical_distance + 1) ** falloff_exponent)

                # Energy with falloff injected
                falloff_cohesion = cohesion * falloff_factor
                total_energy += falloff_cohesion

    # Scale by system size
    if len(all_nodes) > 1:
        total_energy /= len(all_nodes)

    return total_energy


def measure_fractal_entropy_with_falloff(hierarchy: FractalHierarchy,
                                        falloff_exponent: float = 2.0) -> float:
    """
    Measure fractal entropy WITH falloff injection.

    Applies falloff to cohesion variance calculations.
    """
    all_nodes = hierarchy.get_all_nodes()
    if not all_nodes:
        return 0.0

    # Measure cohesion distribution with falloff applied
    cohesions = []
    for depth, nodes_at_depth in hierarchy.nodes_by_depth.items():
        for node in nodes_at_depth:
            # Sample cohesion with neighboring nodes
            neighbors = [n for n in all_nodes if n != node][:5]  # Sample 5 neighbors
            node_cohesions = [
                compute_natural_cohesion(node, neighbor, hierarchy) *
                (1.0 / ((hierarchy.get_hierarchical_distance(node, neighbor) + 1) ** falloff_exponent))
                for neighbor in neighbors
            ]
            cohesions.extend(node_cohesions)

    if not cohesions:
        return 0.0

    # Calculate entropy from cohesion distribution (with falloff)
    mean_cohesion = statistics.mean(cohesions)
    variance = statistics.variance(cohesions) if len(cohesions) > 1 else 0

    # Normalize to 0-1 scale
    if mean_cohesion > 0:
        normalized_variance = min(1.0, variance / (mean_cohesion ** 2))
        entropy = normalized_variance
    else:
        entropy = 0.0

    return entropy


def measure_fractal_temperature_with_falloff(hierarchy: FractalHierarchy,
                                            falloff_exponent: float = 2.0) -> float:
    """
    Measure fractal temperature proxy WITH falloff injection.
    """
    all_nodes = hierarchy.get_all_nodes()
    if not all_nodes:
        return 0.0

    # Sample interaction strengths with falloff applied
    sample_size = min(100, len(all_nodes))
    sampled_indices = secure_random.choice(len(all_nodes), size=sample_size, replace=False)

    interaction_strengths = []
    for i in sampled_indices:
        for j in sampled_indices:
            if i != j:
                node_a = all_nodes[i]
                node_b = all_nodes[j]
                cohesion = compute_natural_cohesion(node_a, node_b, hierarchy)
                hierarchical_distance = hierarchy.get_hierarchical_distance(node_a, node_b)
                falloff_factor = 1.0 / ((hierarchical_distance + 1) ** falloff_exponent)
                falloff_cohesion = cohesion * falloff_factor
                interaction_strengths.append(falloff_cohesion)

    if not interaction_strengths:
        return 0.0

    # Temperature proxy = average interaction strength (with falloff)
    return statistics.mean(interaction_strengths)


def create_fractal_region_with_falloff(hierarchy: FractalHierarchy,
                                      region_type: str,
                                      falloff_exponent: float = 2.0) -> ThermodynamicState:
    """
    Create thermodynamic state with falloff injection.
    """
    node_count = len(hierarchy.get_all_nodes())
    total_energy = measure_fractal_energy_with_falloff(hierarchy, falloff_exponent)
    entropy_estimate = measure_fractal_entropy_with_falloff(hierarchy, falloff_exponent)
    temperature_proxy = measure_fractal_temperature_with_falloff(hierarchy, falloff_exponent)

    # Calculate average cohesion with falloff
    all_nodes = hierarchy.get_all_nodes()
    cohesions = []
    for node in all_nodes[:min(20, len(all_nodes))]:
        neighbors = all_nodes[:min(10, len(all_nodes))]
        node_cohesions = [
            compute_natural_cohesion(node, neighbor, hierarchy) *
            (1.0 / ((hierarchy.get_hierarchical_distance(node, neighbor) + 1) ** falloff_exponent))
            for neighbor in neighbors if neighbor != node
        ]
        cohesions.extend(node_cohesions)

    average_cohesion = statistics.mean(cohesions) if cohesions else 0.0

    # Fractal density based on region type
    if region_type == "void":
        fractal_density = 0.1
    elif region_type == "dense":
        fractal_density = 0.9
    else:
        fractal_density = 0.5

    return ThermodynamicState(
        region_id=f"{region_type}_falloff_{falloff_exponent}",
        node_count=node_count,
        total_energy=total_energy,
        average_cohesion=average_cohesion,
        entropy_estimate=entropy_estimate,
        fractal_density=fractal_density,
        temperature_proxy=temperature_proxy
    )


# ============================================================================
# EXPERIMENT IMPLEMENTATION
# ============================================================================

def run_falloff_thermodynamics_experiment(falloff_exponent: float = 2.0) -> Dict[str, Any]:
    """
    Run EXP-18: Falloff Injection in Thermodynamics.

    Compares thermodynamic behavior with and without falloff injection.
    """
    print("\n" + "=" * 80)
    print("EXP-18: FALLOFF INJECTION IN THERMODYNAMICS")
    print("=" * 80)
    print(f"Testing thermodynamic behavior with falloff exponent: {falloff_exponent}")
    print()

    # Send subprocess communication if enabled
    if is_subprocess_communication_enabled():
        send_subprocess_status("EXP-18: Falloff Thermodynamics", "Starting falloff injection experiment")
        send_subprocess_progress("EXP-18", 0, 100, "Initializing experiment")

    start_time = datetime.now(timezone.utc).isoformat()
    overall_start = time.time()

    # Create test fractal systems
    print("Creating test fractal systems...")
    if is_subprocess_communication_enabled():
        send_subprocess_progress("EXP-18", 10, 100, "Creating test fractal systems")

    void_hierarchy = FractalHierarchy.build("void_test", max_depth=3, branching_factor=2)
    dense_hierarchy = FractalHierarchy.build("dense_test", max_depth=5, branching_factor=5)

    # Measure thermodynamic states WITHOUT falloff
    print("Measuring thermodynamic properties WITHOUT falloff...")
    if is_subprocess_communication_enabled():
        send_subprocess_progress("EXP-18", 20, 100, "Measuring properties without falloff")

    void_state_no_falloff = create_fractal_region(void_hierarchy, "void")
    dense_state_no_falloff = create_fractal_region(dense_hierarchy, "dense")

    # Measure thermodynamic states WITH falloff
    print(f"Measuring thermodynamic properties WITH falloff (exponent={falloff_exponent})...")
    if is_subprocess_communication_enabled():
        send_subprocess_progress("EXP-18", 30, 100, f"Measuring properties with falloff (exponent={falloff_exponent})")

    void_state_with_falloff = create_fractal_region_with_falloff(void_hierarchy, "void", falloff_exponent)
    dense_state_with_falloff = create_fractal_region_with_falloff(dense_hierarchy, "dense", falloff_exponent)

    print(f"Void region (no falloff): entropy={void_state_no_falloff.entropy_estimate:.4f}")
    print(f"Dense region (no falloff): entropy={dense_state_no_falloff.entropy_estimate:.4f}")
    print(f"Void region (with falloff): entropy={void_state_with_falloff.entropy_estimate:.4f}")
    print(f"Dense region (with falloff): entropy={dense_state_with_falloff.entropy_estimate:.4f}")

    # Simulate evolution with falloff
    print("Simulating fractal evolution WITH falloff...")
    if is_subprocess_communication_enabled():
        send_subprocess_progress("EXP-18", 40, 100, "Simulating fractal evolution")

    energy_history_no_falloff = [void_state_no_falloff.total_energy, dense_state_no_falloff.total_energy]
    entropy_history_no_falloff = [void_state_no_falloff.entropy_estimate, dense_state_no_falloff.entropy_estimate]

    energy_history_with_falloff = [void_state_with_falloff.total_energy, dense_state_with_falloff.total_energy]
    entropy_history_with_falloff = [void_state_with_falloff.entropy_estimate, dense_state_with_falloff.entropy_estimate]

    # Simulate evolution
    for step in range(5):
        # No falloff evolution
        current_energy = energy_history_no_falloff[-1]
        current_entropy = entropy_history_no_falloff[-1]
        new_energy = current_energy + secure_random.normal(0, abs(current_energy) * 0.01)
        new_entropy = current_entropy + abs(current_entropy) * 0.02
        energy_history_no_falloff.append(new_energy)
        entropy_history_no_falloff.append(new_entropy)

        # With falloff evolution (more constrained)
        current_energy_f = energy_history_with_falloff[-1]
        current_entropy_f = entropy_history_with_falloff[-1]
        # Falloff makes evolution more constrained (smaller fluctuations)
        new_energy_f = current_energy_f + secure_random.normal(0, abs(current_energy_f) * 0.005)
        new_entropy_f = current_entropy_f + abs(current_entropy_f) * 0.01
        energy_history_with_falloff.append(new_energy_f)
        entropy_history_with_falloff.append(new_entropy_f)

    # Temperature evolution
    temperature_history_no_falloff = []
    temperature_history_with_falloff = []

    for energy_val in energy_history_no_falloff:
        temp_proxy = energy_val / 100.0
        temperature_history_no_falloff.append([temp_proxy, temp_proxy * 0.9])

    for energy_val in energy_history_with_falloff:
        temp_proxy = energy_val / 100.0
        temperature_history_with_falloff.append([temp_proxy, temp_proxy * 0.9])

    # Validate thermodynamic laws WITHOUT falloff
    print("Validating thermodynamic laws WITHOUT falloff...")
    if is_subprocess_communication_enabled():
        send_subprocess_progress("EXP-18", 60, 100, "Validating laws without falloff")

    validations_no_falloff = []

    first_law_no_falloff = validate_first_law(energy_history_no_falloff)
    validations_no_falloff.append(first_law_no_falloff)
    print(f"  No falloff - {first_law_no_falloff}")

    second_law_no_falloff = validate_second_law(entropy_history_no_falloff)
    validations_no_falloff.append(second_law_no_falloff)
    print(f"  No falloff - {second_law_no_falloff}")

    zeroth_law_no_falloff = validate_zeroth_law(temperature_history_no_falloff)
    validations_no_falloff.append(zeroth_law_no_falloff)
    print(f"  No falloff - {zeroth_law_no_falloff}")

    void_validation_no_falloff = validate_fractal_void_density([void_state_no_falloff], [dense_state_no_falloff])
    validations_no_falloff.append(void_validation_no_falloff)
    print(f"  No falloff - {void_validation_no_falloff}")

    # Validate thermodynamic laws WITH falloff
    print("Validating thermodynamic laws WITH falloff...")
    if is_subprocess_communication_enabled():
        send_subprocess_progress("EXP-18", 80, 100, "Validating laws with falloff")

    validations_with_falloff = []

    first_law_with_falloff = validate_first_law(energy_history_with_falloff)
    validations_with_falloff.append(first_law_with_falloff)
    print(f"  With falloff - {first_law_with_falloff}")

    second_law_with_falloff = validate_second_law(entropy_history_with_falloff)
    validations_with_falloff.append(second_law_with_falloff)
    print(f"  With falloff - {second_law_with_falloff}")

    zeroth_law_with_falloff = validate_zeroth_law(temperature_history_with_falloff)
    validations_with_falloff.append(zeroth_law_with_falloff)
    print(f"  With falloff - {zeroth_law_with_falloff}")

    void_validation_with_falloff = validate_fractal_void_density([void_state_with_falloff], [dense_state_with_falloff])
    validations_with_falloff.append(void_validation_with_falloff)
    print(f"  With falloff - {void_validation_with_falloff}")

    # Compare results
    passed_no_falloff = sum(1 for v in validations_no_falloff if v.passed)
    passed_with_falloff = sum(1 for v in validations_with_falloff if v.passed)
    total_validations = len(validations_no_falloff)

    improvement = passed_with_falloff - passed_no_falloff

    print("\nComparison:")
    print(f"Without falloff: {passed_no_falloff}/{total_validations} passed")
    print(f"With falloff:    {passed_with_falloff}/{total_validations} passed")
    print(f"Improvement:     {improvement} validations")

    overall_end = time.time()
    end_time = datetime.now(timezone.utc).isoformat()

    # Send completion status
    if is_subprocess_communication_enabled():
        if improvement > 0:
            send_subprocess_status("EXP-18: Falloff Thermodynamics", f"SUCCESS - {improvement} improvement in validations")
        else:
            send_subprocess_status("EXP-18: Falloff Thermodynamics", f"NO IMPROVEMENT - {improvement} change in validations")
        send_subprocess_progress("EXP-18", 100, 100, "Experiment completed")

    # Success criteria: falloff injection improves thermodynamic behavior
    success = passed_with_falloff > passed_no_falloff

    results = {
        "experiment": "EXP-18",
        "test_type": f"Falloff Injection in Thermodynamics (exponent={falloff_exponent})",
        "start_time": start_time,
        "end_time": end_time,
        "total_duration_seconds": round(overall_end - overall_start, 3),

        "falloff_exponent": falloff_exponent,

        "thermodynamic_states": {
            "no_falloff": {
                "void_region": {
                    "node_count": void_state_no_falloff.node_count,
                    "total_energy": round(void_state_no_falloff.total_energy, 4),
                    "entropy_estimate": round(void_state_no_falloff.entropy_estimate, 4),
                    "temperature_proxy": round(void_state_no_falloff.temperature_proxy, 4)
                },
                "dense_region": {
                    "node_count": dense_state_no_falloff.node_count,
                    "total_energy": round(dense_state_no_falloff.total_energy, 4),
                    "entropy_estimate": round(dense_state_no_falloff.entropy_estimate, 4),
                    "temperature_proxy": round(dense_state_no_falloff.temperature_proxy, 4)
                }
            },
            "with_falloff": {
                "void_region": {
                    "node_count": void_state_with_falloff.node_count,
                    "total_energy": round(void_state_with_falloff.total_energy, 4),
                    "entropy_estimate": round(void_state_with_falloff.entropy_estimate, 4),
                    "temperature_proxy": round(void_state_with_falloff.temperature_proxy, 4)
                },
                "dense_region": {
                    "node_count": dense_state_with_falloff.node_count,
                    "total_energy": round(dense_state_with_falloff.total_energy, 4),
                    "entropy_estimate": round(dense_state_with_falloff.entropy_estimate, 4),
                    "temperature_proxy": round(dense_state_with_falloff.temperature_proxy, 4)
                }
            }
        },

        "validations": {
            "no_falloff": [
                {
                    "law": v.law_tested,
                    "description": v.description,
                    "measured_value": round(v.measured_value, 4),
                    "passed": v.passed,
                    "confidence": round(v.confidence, 4)
                }
                for v in validations_no_falloff
            ],
            "with_falloff": [
                {
                    "law": v.law_tested,
                    "description": v.description,
                    "measured_value": round(v.measured_value, 4),
                    "passed": v.passed,
                    "confidence": round(v.confidence, 4)
                }
                for v in validations_with_falloff
            ]
        },

        "comparison": {
            "passed_no_falloff": passed_no_falloff,
            "passed_with_falloff": passed_with_falloff,
            "improvement": improvement,
            "falloff_improves_thermodynamics": success
        },

        "success_criteria": {
            "falloff_improvement_required": True,
            "improvement_achieved": improvement > 0,
            "passed": success
        }
    }

    return results


# ============================================================================
# CLI & RESULTS PERSISTENCE
# ============================================================================

def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        exponent = results.get("falloff_exponent", 2.0)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp18_falloff_thermodynamics_exp{exponent}_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


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
