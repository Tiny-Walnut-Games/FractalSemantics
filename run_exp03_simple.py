#!/usr/bin/env python3
"""
Simple launcher for EXP-03: Coordinate Entropy Test
This script provides a basic interface to run the experiment.
"""

import sys
import os

# Add the fractalstat directory to Python path
fractalstat_dir = os.path.join(os.path.dirname(__file__), 'fractalstat')
sys.path.insert(0, fractalstat_dir)

def main():
    """Run EXP-03 with default settings."""
    print("Starting EXP-03: Coordinate Entropy Test")
    print("Sample size: 100,000 coordinates per dimension")
    print()
    
    try:
        # Import the experiment module
        from exp03_coordinate_entropy import EXP03_CoordinateEntropy
        
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
            print(f"   Total scales tested: {summary['total_scales_tested']}")
            print(f"   All tests passed: {summary['all_passed']}")
            print(f"   Performance targets met: {summary['performance_targets']}")
            print(f"   Query pattern distribution: {summary['query_pattern_distribution']}")
            
        else:
            print("❌ Experiment failed!")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("The experiment module may have syntax errors.")
        print("Please check the experiment files for issues.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()