#!/usr/bin/env python3
"""
EXP-02: Retrieval Efficiency Test
Direct execution entry point for easy command-line usage.
"""

import sys
import argparse
import os
import json

# Import the main experiment class AFTER path is set up
from experiment import EXP02_RetrievalEfficiency

# Add the current directory to Python path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Add parent directory to path to access fractalstat module
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)
# Add the fractalstat directory to path
fractalstat_dir = os.path.join(parent_dir, "..")
sys.path.insert(0, fractalstat_dir)


def main():
    """Main entry point for EXP-02 experiment."""
    parser = argparse.ArgumentParser(
        description="EXP-02: Retrieval Efficiency Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fractalstat/exp02_retrieval_efficiency/          # Run with default settings
  python fractalstat/exp02_retrieval_efficiency/ --quick  # Quick test with reduced parameters
  python fractalstat/exp02_retrieval_efficiency/ --full   # Full test with maximum parameters
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick test with reduced parameters'
    )
    
    parser.add_argument(
        '--stress', 
        action='store_true',
        help='Stress test with increased parameters'
    )
    
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Full test with maximum parameters'
    )
    
    parser.add_argument(
        '--memories', 
        type=int,
        help='Custom number of memories (overrides quick/stress/full flags)'
    )
    
    args = parser.parse_args()
    
    # Determine parameters based on flags
    if args.quick:
        num_memories = 1000
    elif args.stress:
        num_memories = 10000
    elif args.full:
        num_memories = 50000
    elif args.memories:
        num_memories = args.memories
    else:
        num_memories = 5000  # Default
        
    try:
        # Create and run experiment
        experiment = EXP02_RetrievalEfficiency(query_count=num_memories)
        results_list, success = experiment.run()
        summary = experiment.get_summary()
        
        # Print results summary
        print(f"\n{'='*60}")
        print("EXP-02 RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {'PASS' if success else 'FAIL'}")
        print(f"Total Scales Tested: {summary['total_scales_tested']}")
        print(f"All Tests Passed: {summary['all_passed']}")
        
        # Save results to JSON file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"exp02_retrieval_efficiency_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(current_dir, "..", "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, output_file)
        
        # Save results - convert any non-serializable objects to dict
        def make_serializable(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        serializable_summary = make_serializable(summary)
        with open(output_path, "w") as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Return appropriate exit code
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())