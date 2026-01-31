#!/usr/bin/env python3
"""
EXP-06: Entanglement Detection - Friendly Launcher

This script provides a user-friendly interface for running the entanglement detection experiment.
It follows the same consistent pattern as EXP-01, EXP-02, EXP-03, and EXP-04 launchers.

Usage:
    python run_exp06.py [mode] [threshold]

Examples:
    python run_exp06.py                    # Quick mode (1000 bit-chains, threshold 0.85)
    python run_exp06.py quick              # Quick mode (1000 bit-chains, threshold 0.85)
    python run_exp06.py full               # Full mode (5000 bit-chains, threshold 0.85)
    python run_exp06.py custom 1000 0.90   # Custom mode (1000 bit-chains, threshold 0.90)
    python run_exp06.py test               # Test mode (100 bit-chains, threshold 0.85)

Author: FractalSemantics
Date: 2025-12-07
"""

import sys
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

# Import the experiment module
from fractalstat.exp06_entanglement_detection import run_experiment, save_results


def print_experiment_header(mode: str, sample_size: int, threshold: float):
    """Print experiment header information."""
    print("\n" + "=" * 80)
    print("EXP-06: ENTANGLEMENT DETECTION VALIDATION")
    print("=" * 80)
    print(f"Mode: {mode.title()} ({sample_size:,} bit-chains)")
    print(f"Detection Threshold: {threshold}")
    print()


def print_results_summary(results: Dict[str, Any], success: bool):
    """Print detailed results summary."""
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Basic statistics
    print(f"Sample Size: {results['sample_size']:,} bit-chains")
    print(f"Total Possible Pairs: {results['total_possible_pairs']:,}")
    print(f"True Entangled Pairs: {results['true_entangled_pairs']:,}")
    print(f"Detected Pairs: {results['detected_pairs']:,}")
    print()
    
    # Validation metrics
    metrics = results['validation_metrics']
    print("VALIDATION METRICS:")
    print(f"  Precision: {metrics['precision']:.4f} ({'PASS' if metrics['precision'] >= 0.70 else 'FAIL'})")
    print(f"  Recall: {metrics['recall']:.4f} ({'PASS' if metrics['recall'] >= 0.60 else 'FAIL'})")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.6f}")
    print(f"  Runtime: {metrics['runtime_seconds']:.2f} seconds")
    print()
    
    # Entanglement analysis
    analysis = results['entanglement_analysis']
    print("ENTANGLEMENT ANALYSIS:")
    print(f"  Entanglement Density: {analysis['entanglement_density']:.4f}")
    print(f"  Average Score: {analysis['average_entanglement_score']:.4f}")
    print(f"  Quality Assessment: {analysis['quality_assessment']}")
    print(f"  Cross-Realm Ratio: {analysis['cross_realm_ratio']:.4f}")
    print()
    
    # Score distribution
    score_dist = results['score_distribution']
    print("SCORE DISTRIBUTION:")
    print(f"  Count: {score_dist['count']:,}")
    print(f"  Min: {score_dist['min']:.4f}")
    print(f"  Max: {score_dist['max']:.4f}")
    print(f"  Mean: {score_dist['mean']:.4f}")
    print(f"  Median: {score_dist['median']:.4f}")
    print(f"  Std Dev: {score_dist['std_dev']:.4f}")
    print()
    
    # Success status
    print("=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80)
    if success:
        print("✅ [PASS] ENTANGLEMENT DETECTION: VALIDATED")
        print("   * Algorithm successfully detects semantic relationships")
        print("   * Precision and recall meet performance targets")
        print("   * Quantum identity preservation validated")
    else:
        print("❌ [FAIL] ENTANGLEMENT DETECTION: NEEDS IMPROVEMENT")
        print("   * Algorithm performance below validation targets")
        print("   * Consider adjusting threshold or algorithm parameters")
    print()


def print_usage():
    """Print usage information."""
    print("Usage: python run_exp06.py [mode] [threshold]")
    print()
    print("Modes:")
    print("  quick   - Quick test (1,000 bit-chains, threshold 0.85)")
    print("  full    - Full test (5,000 bit-chains, threshold 0.85)")
    print("  custom  - Custom parameters (requires sample_size and threshold)")
    print("  test    - Test mode (100 bit-chains, threshold 0.85)")
    print()
    print("Examples:")
    print("  python run_exp06.py                    # Quick mode")
    print("  python run_exp06.py quick              # Quick mode")
    print("  python run_exp06.py full               # Full mode")
    print("  python run_exp06.py custom 1000 0.90   # Custom mode")
    print("  python run_exp06.py test               # Test mode")
    print()


def main():
    """Main entry point for EXP-06 launcher."""
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Default: quick mode
        mode = "quick"
        sample_size = 1000
        threshold = 0.85
    elif len(sys.argv) == 2:
        mode = sys.argv[1].lower()
        if mode == "quick":
            sample_size = 1000
            threshold = 0.85
        elif mode == "full":
            sample_size = 5000
            threshold = 0.85
        elif mode == "test":
            sample_size = 100
            threshold = 0.85
        else:
            print("❌ Invalid mode. Use: quick, full, test, or custom")
            print_usage()
            sys.exit(1)
    elif len(sys.argv) == 3:
        mode = sys.argv[1].lower()
        if mode == "custom":
            try:
                sample_size = int(sys.argv[2])
                threshold = 0.85  # Default threshold
            except ValueError:
                print("❌ Invalid sample size. Must be a number.")
                print_usage()
                sys.exit(1)
        else:
            print("❌ Invalid arguments. Use: python run_exp06.py [mode] [threshold]")
            print_usage()
            sys.exit(1)
    elif len(sys.argv) == 4:
        mode = sys.argv[1].lower()
        if mode == "custom":
            try:
                sample_size = int(sys.argv[2])
                threshold = float(sys.argv[3])
            except ValueError:
                print("❌ Invalid parameters. Sample size must be integer, threshold must be float.")
                print_usage()
                sys.exit(1)
        else:
            print("❌ Invalid arguments. Use: python run_exp06.py [mode] [threshold]")
            print_usage()
            sys.exit(1)
    else:
        print("❌ Invalid number of arguments.")
        print_usage()
        sys.exit(1)
    
    # Validate parameters
    if sample_size < 10:
        print("❌ Sample size must be at least 10.")
        sys.exit(1)
    
    if not (0.0 <= threshold <= 1.0):
        print("❌ Threshold must be between 0.0 and 1.0.")
        sys.exit(1)
    
    # Print header
    print_experiment_header(mode, sample_size, threshold)
    
    # Run experiment
    print("Running entanglement detection experiment...")
    start_time = time.time()
    
    try:
        results, success = run_experiment(sample_size=sample_size, threshold=threshold)
        runtime = time.time() - start_time
        
        # Print results
        print_results_summary(results, success)
        
        # Save results
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp06_entanglement_detection_{timestamp}.json"
        save_results(results, output_file)
        
        # Final summary
        print("✅ Experiment completed successfully!")
        print(f"   Tested {sample_size:,} bit-chains")
        print(f"   Runtime: {runtime:.2f} seconds")
        print(f"   Results saved to: {output_file}")
        
        if success:
            print("   Status: [PASS] - Entanglement detection validated")
        else:
            print("   Status: [FAIL] - Performance below targets")
        
        return success
        
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