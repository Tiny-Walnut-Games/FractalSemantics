#!/usr/bin/env python3
"""
FractalStat module entry point.

This module provides the main interface for running FractalStat experiments
and utilities. It can be executed directly as a module or imported as a package.
"""

import sys
import argparse
from pathlib import Path

# Add the current directory to Python path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for FractalStat module."""
    parser = argparse.ArgumentParser(
        description="FractalStat: Self-Organizing Memory Networks and Fractal Coordinate System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fractalstat exp08 --num-memories 1000
  python -m fractalstat exp08 --quick
  python -m fractalstat exp08 --full
        """
    )
    
    parser.add_argument(
        "experiment",
        choices=["exp08", "self-organizing-memory"],
        help="Experiment to run"
    )
    
    parser.add_argument(
        "--num-memories",
        type=int,
        default=1000,
        help="Number of memories to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--consolidation-threshold",
        type=float,
        default=0.8,
        help="Consolidation threshold (default: 0.8)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with 100 memories"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test with 5000 memories"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_memories = 100
    elif args.full:
        args.num_memories = 5000
    
    try:
        if args.experiment in ["exp08", "self-organizing-memory"]:
            from fractalstat.exp08_self_organizing_memory.experiment import SelfOrganizingMemoryExperiment
            
            print(f"Running {args.experiment} with {args.num_memories} memories...")
            
            experiment = SelfOrganizingMemoryExperiment(
                num_memories=args.num_memories,
                consolidation_threshold=args.consolidation_threshold
            )
            
            results = experiment.run()
            
            # Print summary
            print("\n" + "=" * 60)
            print("EXPERIMENT SUMMARY")
            print("=" * 60)
            print(f"Status: {results.status}")
            print(f"Total Memories: {results.total_memories}")
            print(f"Number of Clusters: {results.num_clusters}")
            print(f"Semantic Cohesion: {results.semantic_cohesion_score:.3f}")
            print(f"Retrieval Efficiency: {results.retrieval_efficiency:.3f}")
            print(f"Storage Reduction: {results.storage_overhead_reduction:.3f}")
            print(f"Emergent Intelligence: {results.emergent_intelligence_score:.3f}")
            
            return results.status == "PASS"
        
        else:
            print(f"Unknown experiment: {args.experiment}")
            return False
            
    except ImportError as e:
        print(f"Error importing experiment: {e}")
        print("Make sure the experiment modules are properly installed.")
        return False
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)