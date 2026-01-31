#!/usr/bin/env python3
"""
EXP-07: LUCA Bootstrap Test - Friendly Launcher

This script provides a user-friendly interface for running the LUCA bootstrap test.
It includes clear progress indicators, detailed results, and easy-to-understand output.

Usage:
    python run_exp07.py [mode] [num_entities]

Examples:
    python run_exp07.py                    # Quick mode (10 entities)
    python run_exp07.py quick              # Quick mode (10 entities)
    python run_exp07.py full               # Full mode (1000 entities)
    python run_exp07.py custom 100         # Custom mode (100 entities)
    python run_exp07.py test               # Test mode (5 entities)

Author: FractalSemantics
Date: 2025-12-07
"""

import sys
import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import the experiment module
from fractalstat.exp07_luca_bootstrap import run_experiment_from_config, save_results


def print_experiment_header(mode: str, num_entities: int):
    """Print experiment header information."""
    print("\n" + "=" * 80)
    print("EXP-07: LUCA BOOTSTRAP TEST")
    print("=" * 80)
    print(f"Mode: {mode.title()} ({num_entities:,} entities)")
    print("Testing: Can we compress system to LUCA and reconstruct perfectly?")
    print()


def print_results_summary(results: Dict[str, Any], success: bool):
    """Print detailed results summary."""
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Basic statistics
    print(f"Tested Entities: {results['bootstrap']['bootstrapped_count']:,}")
    print(f"Compression Ratio: {results['compression']['ratio']:.2f}x")
    print(f"Original Size: {results['compression']['original_size']:,} bytes")
    print(f"LUCA Size: {results['compression']['luca_size']:,} bytes")
    print()
    
    # Recovery rates
    recovery = results['comparison']
    print("RECOVERY RATES:")
    print(f"  Entity Recovery: {recovery['entity_recovery_rate']:.1%} ({'PASS' if recovery['entity_recovery_rate'] >= 1.0 else 'FAIL'})")
    print(f"  Lineage Recovery: {recovery['lineage_recovery_rate']:.1%} ({'PASS' if recovery['lineage_recovery_rate'] >= 1.0 else 'FAIL'})")
    print(f"  Realm Recovery: {recovery['realm_recovery_rate']:.1%} ({'PASS' if recovery['realm_recovery_rate'] >= 1.0 else 'FAIL'})")
    print(f"  Dimensionality Recovery: {recovery['dimensionality_recovery_rate']:.1%} ({'PASS' if recovery['dimensionality_recovery_rate'] >= 1.0 else 'FAIL'})")
    print(f"  Information Loss: {'YES' if recovery['information_loss'] else 'NO'} ({'PASS' if not recovery['information_loss'] else 'FAIL'})")
    print()
    
    # Fractal properties
    fractal = results['fractal']
    print("FRACTAL PROPERTIES:")
    print(f"  Self-Similarity: {fractal['self_similarity']} ({'PASS' if fractal['self_similarity'] else 'FAIL'})")
    print(f"  Scale Invariance: {fractal['scale_invariance']} ({'PASS' if fractal['scale_invariance'] else 'FAIL'})")
    print(f"  Recursive Structure: {fractal['recursive_structure']} ({'PASS' if fractal['recursive_structure'] else 'FAIL'})")
    print(f"  LUCA Traceability: {fractal['luca_traceability']} ({'PASS' if fractal['luca_traceability'] else 'FAIL'})")
    print(f"  Fractal Score: {fractal['fractal_score']:.2f} ({'PASS' if fractal['fractal_score'] >= 0.8 else 'FAIL'})")
    print()
    
    # Continuity testing
    continuity = results['continuity']
    print("CONTINUITY TESTING:")
    print(f"  Bootstrap Cycles: {continuity['cycles_performed']}")
    print(f"  Bootstrap Failures: {continuity['failures']}")
    print(f"  Lineage Continuity: {continuity['lineage_preserved']} ({'PASS' if continuity['lineage_preserved'] else 'FAIL'})")
    print()
    
    # Success status
    print("=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80)
    if success:
        print("✅ [PASS] LUCA BOOTSTRAP: PERFECT RECONSTRUCTION ACHIEVED!")
        print("   * System can be compressed to LUCA and fully reconstructed")
        print("   * All entities recovered with 100% accuracy")
        print("   * Fractal properties preserved through compression/expansion")
        print("   * Multiple bootstrap cycles completed successfully")
    else:
        print("❌ [FAIL] LUCA BOOTSTRAP: NEEDS IMPROVEMENT")
        print("   * System reconstruction failed validation criteria")
        print("   * Check recovery rates and fractal property preservation")
    print()


def print_usage():
    """Print usage information."""
    print("Usage: python run_exp07.py [mode] [num_entities]")
    print()
    print("Modes:")
    print("  quick   - Quick test (10 entities)")
    print("  full    - Full test (1000 entities)")
    print("  custom  - Custom number of entities")
    print("  test    - Test mode (5 entities)")
    print()
    print("Examples:")
    print("  python run_exp07.py                    # Quick mode")
    print("  python run_exp07.py quick              # Quick mode")
    print("  python run_exp07.py full               # Full mode")
    print("  python run_exp07.py custom 100         # Custom mode")
    print("  python run_exp07.py test               # Test mode")
    print()


def main():
    """Main entry point for EXP-07 launcher."""
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Default: quick mode
        mode = "quick"
        num_entities = 10
    elif len(sys.argv) == 2:
        mode = sys.argv[1].lower()
        if mode == "quick":
            num_entities = 10
        elif mode == "full":
            num_entities = 1000
        elif mode == "test":
            num_entities = 5
        else:
            print("❌ Invalid mode. Use: quick, full, test, or custom")
            print_usage()
            sys.exit(1)
    elif len(sys.argv) == 3:
        mode = sys.argv[1].lower()
        if mode == "custom":
            try:
                num_entities = int(sys.argv[2])
            except ValueError:
                print("❌ Invalid number of entities. Must be an integer.")
                print_usage()
                sys.exit(1)
        else:
            print("❌ Invalid arguments. Use: python run_exp07.py [mode] [num_entities]")
            print_usage()
            sys.exit(1)
    else:
        print("❌ Invalid number of arguments.")
        print_usage()
        sys.exit(1)
    
    # Validate parameters
    if num_entities < 1:
        print("❌ Number of entities must be at least 1.")
        sys.exit(1)
    
    if num_entities > 1000000:
        print("❌ Number of entities cannot exceed 1,000,000.")
        sys.exit(1)
    
    # Print header
    print_experiment_header(mode, num_entities)
    
    # Run experiment
    print("Running LUCA bootstrap test...")
    start_time = time.time()
    
    try:
        # Configure experiment
        config = {
            "num_entities": num_entities
        }
        
        # Run experiment
        results = run_experiment_from_config(config)
        runtime = time.time() - start_time
        
        # Print results
        print_results_summary(results.results, results.status == "PASS")
        
        # Save results
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp07_luca_bootstrap_{timestamp}.json"
        save_results(results.to_dict(), output_file)
        
        # Final summary
        print("✅ Experiment completed successfully!")
        print(f"   Tested {num_entities:,} entities")
        print(f"   Runtime: {runtime:.2f} seconds")
        print(f"   Results saved to: {output_file}")
        
        if results.status == "PASS":
            print("   Status: [PASS] - LUCA bootstrap validated")
        else:
            print("   Status: [FAIL] - Performance below targets")
        
        return results.status == "PASS"
        
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)