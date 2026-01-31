#!/usr/bin/env python3
"""
Simple launcher for EXP-02: Retrieval Efficiency Test
This script respects the modularized structure and uses proper Python imports.
"""

import sys
import os

# Add the fractalstat directory to Python path
fractalstat_dir = os.path.join(os.path.dirname(__file__), 'fractalstat')
sys.path.insert(0, fractalstat_dir)

# Import and run the experiment
from fractalstat.exp02_retrieval_efficiency import EXP02_RetrievalEfficiency
from fractalstat.exp02_retrieval_efficiency.experiment import save_results



def main():
    """Run EXP-02 with default settings."""
    print("Starting EXP-02: Retrieval Efficiency Test")
    print("Sample size: 100,000 coordinates per dimension")
    print()
    
    try:
        # Run the experiment
        experiment = EXP02_RetrievalEfficiency(query_count=100000)
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
            
            # Export results to JSON file            
            save_results(summary)
            
        else:
            print("❌ Experiment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()