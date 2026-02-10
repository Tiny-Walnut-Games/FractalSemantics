"""
FractalSemantics Validation Experiments: Complete Orchestrator

Imports and runs all individual experiment modules (EXP-01 through EXP-12).
Maintains modular architecture where each experiment is a standalone module.

Benefits:
- Modular design: Each experiment can be developed/tested independently
- Reduced complexity: No duplicate experiment code in this orchestrator
- Easier maintenance: Changes to experiments only need to be made in their modules
- Proper import organization: Fixes linter warnings about import placement

Usage:
    python fractalsemantics_experiments.py  # Run all experiments
    python fractalsemantics/exp01_geometric_collision.py  # Run individual experiment
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Re-export functions and classes that moved to separate modules during refactoring
# plus constants from the main package for backward compatibility
try:
    from fractalsemantics.fractalsemantics_entity import (
        FractalSemanticsCoordinates,
        Coordinates,
        BitChain,
        canonical_serialize,
        compute_address_hash,
        generate_random_bitchain,
        REALMS,
        HORIZONS,
        POLARITY_LIST,
        ALIGNMENT_LIST,
        ENTITY_TYPES,
    )
    # Import constants from main package
    from fractalsemantics import POLARITY, ALIGNMENT

    # Import experiment result classes and enums
    from fractalsemantics.exp01_geometric_collision import (
        EXP01_Result,
        EXP01_GeometricCollisionResistance,
    )
    from fractalsemantics.exp02_retrieval_efficiency import (
        EXP02_Result,
        EXP02_RetrievalEfficiency,
    )
    from fractalsemantics.exp03_coordinate_entropy import (
        EXP03_Result,
        EXP03_CoordinateEntropy,
    )
    from fractalsemantics.experiment_utils import (
        normalize_float,
        normalize_timestamp,
        sort_json_keys,
        DataClass,
        Capability,
    )

except ImportError as e:
    print(f"Warning: Some imports failed from fractalsemantics_experiments: {e}")
    # Fallback for missing imports (will cause runtime errors but better than silent failures)
    pass

# Define public interface for backward compatibility
__all__ = [
    # Re-exported from fractalsemantics_entity
    "FractalSemanticsCoordinates",
    "Coordinates",
    "BitChain",
    "canonical_serialize",
    "compute_address_hash",
    "generate_random_bitchain",
    "REALMS",
    "HORIZONS",
    "POLARITY_LIST",
    "ALIGNMENT_LIST",
    "ENTITY_TYPES",
    # Re-exported from main package
    "POLARITY",
    "ALIGNMENT",
    # Re-exported from experiment modules
    "EXP01_Result",
    "EXP01_GeometricCollisionResistance",
    "EXP02_Result",
    "EXP02_RetrievalEfficiency",
    "EXP03_Result",
    "EXP03_CoordinateEntropy",
    # Re-exported from experiment_utils
    "normalize_float",
    "normalize_timestamp",
    "sort_json_keys",
    "DataClass",
    "Capability",
    # Local functions
    "run_single_experiment",
    "run_all_experiments",
    "main",
]


# ===========================================================================
# CONFIGURE EXPERIMENT MODULES TO RUN
# ===========================================================================

# List of all experiments with their module names and display names
EXPERIMENTS = [
    ("exp01_geometric_collision", "EXP-01 (Geometric Collision Resistance)"),
    ("exp02_retrieval_efficiency", "EXP-02 (Retrieval Efficiency)"),
    ("exp03_coordinate_entropy", "EXP-03 (Coordinate Entropy)"),
    ("exp04_fractal_scaling", "EXP-04 (Fractal Scaling)"),
    ("exp05_compression_expansion", "EXP-05 (Compression Expansion)"),
    ("exp06_entanglement_detection", "EXP-06 (Entanglement Detection)"),
    ("exp07_luca_bootstrap", "EXP-07 (LUCA Bootstrap)"),
    ("exp08_self_organizing_memory", "EXP-08 (Self-Organizing Memory Networks)"),
    ("exp09_memory_pressure", "EXP-09 (Memory Pressure)"),
    ("exp10_multidimensional_query", "EXP-10 (Multi-Dimensional Query Optimization)"),
    ("exp11_dimension_cardinality", "EXP-11 (Dimension Cardinality)"),
    ("exp11b_dimension_stress_test", "EXP-11b (Dimension Stress Test)"),
    ("exp12_benchmark_comparison", "EXP-12 (Benchmark Comparison)"),
]


def run_single_experiment(module_name: str, display_name: str) -> Dict[str, Any]:
    """
    Run a single experiment module and capture its results.

    Args:
        module_name: Name of the experiment module (e.g., 'exp01_geometric_collision')
        display_name: Human-readable name for the experiment

    Returns:
        Dictionary with success status and results
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"RUNNING: {display_name}")
        print(f"MODULE: {module_name}")
        print('=' * 80)

        # Try direct execution first (more reliable for individual modules)
        try:
            result = subprocess.run([
                sys.executable, str(Path(__file__).parent / f"{module_name}.py")
            ], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=Path(__file__).parent.parent, check=False)

            # Determine success based on both exit code and output content
            # Check for status indicators in the output
            stdout_text = result.stdout.strip() if result.stdout else ""
            stderr_text = result.stderr.strip() if result.stderr else ""
            
            # Look for status indicators in the output
            success_indicators = ["[OK]", "[Success]", "EXP-01 COMPLETE", "EXP-02 COMPLETE", "EXP-03 COMPLETE", "EXP-04 COMPLETE", "EXP-05 COMPLETE", "EXP-06 COMPLETE", "EXP-07 COMPLETE", "EXP-08 COMPLETE", "EXP-09 COMPLETE", "EXP-10 COMPLETE", "EXP-11 COMPLETE", "EXP-11b COMPLETE", "EXP-12 COMPLETE"]
            failure_indicators = ["[FAIL] EXPERIMENT FAILED", "[Error]", "EXPERIMENT FAILED"]
            
            has_success_indicator = any(indicator in stdout_text for indicator in success_indicators)
            has_failure_indicator = any(indicator in stdout_text for indicator in failure_indicators)
            
            # Determine success
            if has_failure_indicator:
                success = False
            elif has_success_indicator:
                success = True
            else:
                # Fall back to exit code
                success = result.returncode == 0

            return {
                "success": success,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "exit_code": result.returncode
            }
        except Exception as e:
            return {"success": False, "error": f"Direct execution failed: {e}"}

    except ImportError as e:
        error_msg = f"Failed to import {module_name}: {e}"
        print(error_msg)
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error running {module_name}: {e}"
        print(error_msg)
        return {"success": False, "error": error_msg}

    # Fallback return for any unexpected code path
    return {"success": False, "error": f"No handler found for {module_name}"}


def run_all_experiments(selected_experiments: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run all configured experiments or a selected subset.

    Args:
        selected_experiments: List of experiment names to run.
                            If None, runs all experiments.

    Returns:
        Dictionary with experiment results and overall summary
    """
    print("=" * 100)
    print("FRACTALSEMANTICS VALIDATION EXPERIMENTS - MODULAR ORCHESTRATOR")
    print("=" * 100)
    print(f"Total experiments available: {len(EXPERIMENTS)}")
    print()

    if selected_experiments:
        print(f"Running selected experiments: {', '.join(selected_experiments)}")
        experiments_to_run = [exp for exp in EXPERIMENTS if exp[0] in selected_experiments]
    else:
        print("Running all experiments...")
        experiments_to_run = EXPERIMENTS

    print(f"Will run {len(experiments_to_run)} experiment(s)")
    print()

    results = {}
    overall_success = True

    for module_name, display_name in experiments_to_run:
        try:
            exp_result = run_single_experiment(module_name, display_name)
            results[module_name.upper()] = {
                "display_name": display_name,
                "success": exp_result.get("success", False),
                "results": exp_result
            }

            # Update overall success
            if not exp_result.get("success", False):
                overall_success = False

            # Print status
            status = "[PASS]" if exp_result.get("success", False) else "[FAIL]"
            print(f"\n{status}: {display_name}")

        except Exception as e:
            results[module_name.upper()] = {
                "display_name": display_name,
                "success": False,
                "error": str(e)
            }
            overall_success = False
            print(f"\n[FAIL]: {display_name} - {str(e)}")

    # Print final summary
    print(f"\n{'=' * 100}")
    print("EXPERIMENT ORCHESTRATOR SUMMARY")
    print(f"{'=' * 100}")

    successful_experiments = sum(1 for r in results.values() if r["success"])
    total_experiments = len(results)

    for _, exp_data in results.items():
        status_icon = "[PASS]" if exp_data["success"] else "[FAIL]"
        print(f"{status_icon} {exp_data['display_name']}")

    print(f"\nOVERALL RESULT: {'ALL PASS' if overall_success else 'SOME FAILED'}")
    print(f"Successful: {successful_experiments}/{total_experiments}")
    print(f"Success Rate: {(successful_experiments/total_experiments)*100:.1f}%")

    return {
        "overall_success": overall_success,
        "total_experiments": total_experiments,
        "successful_experiments": successful_experiments,
        "results": results
    }


def main():
    """
    Main entry point for running experiments from command line.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run FractalSemantics validation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fractalsemantics_experiments.py                    # Run all experiments
    python fractalsemantics_experiments.py exp01 exp02      # Run specific experiments
    python fractalsemantics_experiments.py --list             # List available experiments
        """
    )

    parser.add_argument(
        'experiments',
        nargs='*',
        help='Specific experiments to run (e.g., exp01 exp02). If none specified, runs all.'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments and exit'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='JSON output file for results'
    )

    args = parser.parse_args()

    if args.list:
        print("Available FractalSemantics Experiments:")
        print("-" * 40)
        for module_name, display_name in EXPERIMENTS:
            print(f"  {module_name:<30} {display_name}")
        print(f"\nTotal: {len(EXPERIMENTS)} experiments")
        return

    # Run experiments
    if args.experiments:
        # Validate experiment names
        invalid_experiments = [exp for exp in args.experiments if exp not in [e[0] for e in EXPERIMENTS]]
        if invalid_experiments:
            print(f"Error: Unknown experiments: {', '.join(invalid_experiments)}")
            print("Use --list to see available experiments")
            sys.exit(1)

        selected_experiments = args.experiments
    else:
        selected_experiments = None

    try:
        results = run_all_experiments(selected_experiments)

        # Save results if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                import json
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

        # Exit with appropriate code
        success = results.get("overall_success", False)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
