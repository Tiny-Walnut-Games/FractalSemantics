#!/usr/bin/env python3
"""
EXP-01: Geometric Collision Resistance Test
Direct execution entry point for easy command-line usage.
"""

import sys
import argparse
import os
import json
import time
from typing import List, Tuple, Dict, Any

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the main experiment class
from experiment import EXP01_GeometricCollisionResistance


def main():
    """Main entry point for EXP-01 experiment."""
    parser = argparse.ArgumentParser(
        description="EXP-01: Geometric Collision Resistance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fractalstat/exp01_geometric_collision/          # Run with default 100k samples
  python fractalstat/exp01_geometric_collision/ --quick  # Quick test with 10k samples
  python fractalstat/exp01_geometric_collision/ --stress # Stress test with 500k samples
  python fractalstat/exp01_geometric_collision/ --max    # Maximum scale test with 1M samples
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick test with 10,000 samples'
    )
    
    parser.add_argument(
        '--stress', 
        action='store_true',
        help='Stress test with 500,000 samples'
    )
    
    parser.add_argument(
        '--max', 
        action='store_true',
        help='Maximum scale test with 1,000,000 samples'
    )
    
    parser.add_argument(
        '--samples', 
        type=int,
        help='Custom sample size (overrides quick/stress/max flags)'
    )
    
    args = parser.parse_args()
    
    # Determine sample size
    if args.samples:
        sample_size = args.samples
    elif args.quick:
        sample_size = 10000
    elif args.stress:
        sample_size = 500000
    elif args.max:
        sample_size = 1000000
    else:
        sample_size = 100000  # Default
    
    print(f"Starting EXP-01: Geometric Collision Resistance Test")
    print(f"Sample size: {sample_size:,} coordinates per dimension")
    print()
    
    try:
        # Run the experiment
        experiment = EXP01_GeometricCollisionResistance(sample_size=sample_size)
        results, success = experiment.run()
        
        if success:
            print()
            print("✅ Experiment completed successfully!")
            print(f"   Tested {len(results)} dimensions")
            
            # Show summary
            summary = experiment.get_summary()
            print()
            print("📊 Summary:")
            print(f"   Low dimension collision rate: {summary['geometric_validation']['low_dimension_collision_rate']}")
            print(f"   High dimension collision rate: {summary['geometric_validation']['high_dimension_collision_rate']}")
            print(f"   Geometric transition confirmed: {summary['geometric_validation']['geometric_transition_confirmed']}")
            
            # Save results
            output_file = save_results(summary)
            print(f"   Results saved to: {output_file}")
            
        else:
            print("❌ Experiment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


def save_results(summary: Dict[str, Any]) -> str:
    """Save experiment results to a JSON file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"exp01_geometric_collision_{timestamp}.json"
    filepath = os.path.join(current_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        return filepath
    except Exception as e:
        print(f"   Warning: Could not save results: {e}")
        return "results not saved"


if __name__ == "__main__":
    main()