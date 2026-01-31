"""
EXP-04: FractalStat Fractal Scaling Test - Main Entry Point

This module provides the main entry point for running the fractal scaling test
as a standalone script or via Python module execution.
"""

import sys
import traceback
from datetime import datetime, timezone

from .experiment import run_fractal_scaling_test
from .results import save_results


def main():
    """Main entry point for EXP-04."""
    # Load from config or fall back to command-line args
    try:
        from fractalstat.config import ExperimentConfig

        exp_config = ExperimentConfig()
        quick_mode = exp_config.get("EXP-04", "quick_mode", True)
    except Exception:
        quick_mode = "--full" not in sys.argv

    try:
        results = run_fractal_scaling_test(quick_mode=quick_mode)
        output_file = save_results(results)

        print("\n" + "=" * 70)
        print("EXP-04 COMPLETE")
        print("=" * 70)
        print(
            f"Status: {
                'PASSED'
                if all(r.is_valid() for r in results.scale_results)
                else 'FAILED'
            }"
        )
        print(f"Fractal: {'YES' if results.is_fractal else 'NO'}")
        print(f"Output: {output_file}")
        print()

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()