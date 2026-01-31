#!/usr/bin/env python3
"""
Simple launcher for EXP-03: Coordinate Entropy Test
This script respects the modularized structure and uses proper Python imports.
"""

import sys
import os

# Add the fractalstat directory to Python path
fractalstat_dir = os.path.join(os.path.dirname(__file__), 'fractalstat')
sys.path.insert(0, fractalstat_dir)

# Import and run the experiment
from exp03_coordinate_entropy import EXP03_CoordinateEntropy


def main():
    """Run EXP-03 with default settings."""
    print("Starting EXP-03: Coordinate Entropy Test")
    print("Sample size: 100,000 coordinates per dimension")
    print()
    
    try:
        # Run the experiment
        experiment = EXP03_CoordinateEntropy(sample_size=100000)
        results, success = experiment.run()
        
        if success:
            print()
            print("✅ Experiment completed successfully!")
            print(f"   Tested {len(results)} dimensions")
            
            # Show summary
            summary = experiment.get_summary()
            print()
            print("📊 Summary:")
            print(f"   Sample size: {summary['sample_size']}")
            print(f"   Random seed: {summary['random_seed']}")
            print(f"   Baseline entropy: {summary['baseline_entropy']:.4f} bits")
            print(f"   All dimensions critical: {summary['all_critical']}")
            print(f"   Total tests: {summary['total_tests']}")
            
            # Save results to JSON file
            from exp03_coordinate_entropy.experiment import save_results
            save_results(summary)
            
        else:
            print("❌ Experiment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()