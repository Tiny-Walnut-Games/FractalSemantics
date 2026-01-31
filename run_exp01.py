#!/usr/bin/env python3
"""
Simple launcher for EXP-01: Geometric Collision Resistance Test
This script respects the modularized structure and uses proper Python imports.
"""

import sys
import os

# Add the fractalstat directory to Python path
fractalstat_dir = os.path.join(os.path.dirname(__file__), 'fractalstat')
sys.path.insert(0, fractalstat_dir)

# Import and run the experiment
from exp01_geometric_collision import EXP01_GeometricCollisionResistance


def main():
    """Run EXP-01 with default settings."""
    print("Starting EXP-01: Geometric Collision Resistance Test")
    print("Sample size: 100,000 coordinates per dimension")
    print()
    
    try:
        # Run the experiment
        experiment = EXP01_GeometricCollisionResistance(sample_size=100000)
        results, success = experiment.run()
        
        if success:
            print()
            print("✅ Experiment completed successfully!")
            print(f"   Tested {len(results)} dimensions")
            
            # Show summary
            summary = experiment.get_summary()
            print()
            print("📊 Summary:")
            print(f"   Low dimension collision rate: {summary['geometric_validation']['low_dimensions_avg_collision_rate']}")
            print(f"   High dimension collision rate: {summary['geometric_validation']['high_dimensions_avg_collision_rate']}")
            print(f"   Geometric transition confirmed: {summary['geometric_validation']['geometric_transition_confirmed']}")
            
            # Save results
            from exp01_geometric_collision.results import save_results
            output_file = save_results(summary)
            print(f"   Results saved to: {output_file}")
            
        else:
            print("❌ Experiment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()