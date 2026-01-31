#!/usr/bin/env python3
"""
Simple launcher for EXP-05: Compression/Expansion Losslessness Validation
This script respects the modularized structure and uses proper Python imports.
"""

import sys
import os

# Add the fractalstat directory to Python path
fractalstat_dir = os.path.join(os.path.dirname(__file__), 'fractalstat')
sys.path.insert(0, fractalstat_dir)

# Import and run the experiment
from exp05_compression_expansion import run_compression_expansion_test, save_results


def main():
    """Run EXP-05 with default settings."""
    print("Starting EXP-05: Compression/Expansion Losslessness Validation")
    print("Mode: Quick (10,000 bit-chains)")
    print()
    
    try:
        # Run the experiment with quick mode
        results = run_compression_expansion_test(num_bitchains=10000, show_samples=True)
        
        print()
        print("✅ Experiment completed successfully!")
        print(f"   Tested {results.num_bitchains_tested:,} bit-chains")
        
        # Show summary
        print()
        print("📊 Summary:")
        print(f"   Start time: {results.start_time}")
        print(f"   End time: {results.end_time}")
        print(f"   Total duration: {results.total_duration_seconds:.1f} seconds")
        print(f"   Lossless system: {'YES' if results.is_lossless else 'NO'}")
        
        # Show compression metrics
        print()
        print("📈 Compression Metrics:")
        print(f"   Average compression ratio: {results.avg_compression_ratio:.3f}x")
        print(f"   Average luminosity decay: {results.avg_luminosity_decay_ratio:.4f}")
        print(f"   Average coordinate accuracy: {results.avg_coordinate_accuracy:.1%}")
        print(f"   Provenance integrity: {results.percent_provenance_intact:.1f}%")
        print(f"   Narrative preservation: {results.percent_narrative_preserved:.1f}%")
        print(f"   Expandability: {results.percent_expandable:.1f}%")
        
        # Show major findings
        print()
        print("🔍 Major Findings:")
        for finding in results.major_findings:
            print(f"   {finding}")
        
        # Save results
        output_file = save_results(results)
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()