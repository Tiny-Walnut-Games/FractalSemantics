#!/usr/bin/env python3
"""
Simple launcher for EXP-04: Fractal Scaling Test
This script respects the modularized structure and uses proper Python imports.
"""

import sys
import os

# Add the fractalstat directory to Python path
fractalstat_dir = os.path.join(os.path.dirname(__file__), 'fractalstat')
sys.path.insert(0, fractalstat_dir)

# Import and run the experiment
from exp04_fractal_scaling import run_fractal_scaling_test, save_results


def main():
    """Run EXP-04 with default settings."""
    print("Starting EXP-04: Fractal Scaling Test")
    print("Mode: Quick (1K, 10K, 100K bit-chains)")
    print()
    
    try:
        # Run the experiment
        results = run_fractal_scaling_test(quick_mode=True)
        
        print()
        print("✅ Experiment completed successfully!")
        print(f"   Tested {len(results.scale_results)} scales")
        
        # Show summary
        print()
        print("📊 Summary:")
        print(f"   Start time: {results.start_time}")
        print(f"   End time: {results.end_time}")
        print(f"   Total duration: {results.total_duration_seconds:.1f} seconds")
        print(f"   All scales valid: {all(r.is_valid() for r in results.scale_results)}")
        print(f"   Is fractal: {results.is_fractal}")
        
        # Show scale-by-scale results
        print()
        print("📈 Scale Results:")
        for i, scale_result in enumerate(results.scale_results):
            print(f"   Scale {i+1} ({scale_result.scale:,} bit-chains):")
            print(f"     Unique addresses: {scale_result.unique_addresses:,}")
            print(f"     Collisions: {scale_result.collision_count} ({scale_result.collision_rate * 100:.2f}%)")
            print(f"     Retrieval time: {scale_result.retrieval_mean_ms:.6f}ms (mean)")
            print(f"     Throughput: {scale_result.addresses_per_second:,.0f} addr/sec")
            print(f"     Valid: {'YES' if scale_result.is_valid() else 'NO'}")
        
        # Show degradation analysis
        print()
        print("🔍 Degradation Analysis:")
        print(f"   Collision: {results.collision_degradation}")
        print(f"   Retrieval: {results.retrieval_degradation}")
        
        # Save results
        output_file = save_results(results)
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()