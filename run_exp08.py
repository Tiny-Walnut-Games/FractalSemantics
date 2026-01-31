#!/usr/bin/env python3
"""
EXP-08: Self-Organizing Memory Networks - User-Friendly Launcher

This script provides an easy-to-use interface for running the Self-Organizing Memory Networks experiment.
It offers multiple execution modes and detailed progress reporting.

Usage:
    python run_exp08.py [mode] [options]

Examples:
    python run_exp08.py quick          # Quick test with 100 memories
    python run_exp08.py standard       # Standard test with 1000 memories
    python run_exp08.py full           # Full test with 5000 memories
    python run_exp08.py custom 2000    # Custom test with 2000 memories
    python run_exp08.py --help         # Show help and options
"""

import sys
import argparse
import time
from pathlib import Path

# Add the current directory to Python path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent))

# Add parent directory to path to access fractalstat module
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header():
    """Print the experiment header."""
    print("=" * 80)
    print("EXP-08: SELF-ORGANIZING MEMORY NETWORKS")
    print("=" * 80)
    print("Investigating emergent intelligence in self-organizing memory systems")
    print("that demonstrate organic growth patterns and semantic clustering.")
    print()

def print_mode_info(mode, num_memories, consolidation_threshold):
    """Print information about the selected mode."""
    print("EXECUTION MODE:", mode.upper())
    print("Memory Count:", num_memories)
    print("Consolidation Threshold:", consolidation_threshold)
    print()

def print_phase_header(phase_name):
    """Print a formatted phase header."""
    print("-" * 60)
    print(f"Phase: {phase_name}")
    print("-" * 60)

def print_results_summary(results):
    """Print a formatted results summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"Status: {results.status}")
    print(f"Total Memories: {results.total_memories}")
    print(f"Number of Clusters: {results.num_clusters}")
    print(f"Semantic Cohesion: {results.semantic_cohesion_score:.3f}")
    print(f"Retrieval Efficiency: {results.retrieval_efficiency:.3f}")
    print(f"Storage Reduction: {results.storage_overhead_reduction:.3f}")
    print(f"Emergent Intelligence: {results.emergent_intelligence_score:.3f}")
    
    if results.status == "PASS":
        print("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("The self-organizing memory network demonstrates:")
        print("  • Organic clustering behavior")
        print("  • Semantic similarity preservation")
        print("  • Emergent intelligence properties")
        print("  • Efficient retrieval mechanisms")
    else:
        print("\n❌ EXPERIMENT FAILED!")
        print("Check the error messages above for details.")

def main():
    """Main entry point for the EXP-08 launcher."""
    parser = argparse.ArgumentParser(
        description="EXP-08: Self-Organizing Memory Networks - User-Friendly Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_exp08.py quick          # Quick test with 100 memories
  python run_exp08.py standard       # Standard test with 1000 memories  
  python run_exp08.py full           # Full test with 5000 memories
  python run_exp08.py custom 2000    # Custom test with 2000 memories
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["quick", "standard", "full", "custom"],
        help="Execution mode: quick (100), standard (1000), full (5000), or custom"
    )
    
    parser.add_argument(
        "num_memories",
        type=int,
        nargs="?",
        default=1000,
        help="Number of memories for custom mode (ignored for other modes)"
    )
    
    parser.add_argument(
        "--consolidation-threshold",
        type=float,
        default=0.8,
        help="Consolidation threshold for clustering (default: 0.8)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress indicators during execution"
    )
    
    args = parser.parse_args()
    
    # Determine memory count based on mode
    if args.mode == "quick":
        num_memories = 100
        mode_name = "Quick Test"
    elif args.mode == "standard":
        num_memories = 1000
        mode_name = "Standard Test"
    elif args.mode == "full":
        num_memories = 5000
        mode_name = "Full Test"
    else:  # custom
        num_memories = args.num_memories
        mode_name = f"Custom Test ({num_memories} memories)"
    
    # Print header and configuration
    print_header()
    print_mode_info(mode_name, num_memories, args.consolidation_threshold)
    
    try:
        # Import the experiment module
        from fractalstat.exp08_self_organizing_memory.experiment import SelfOrganizingMemoryExperiment
        
        # Create and run the experiment
        print_phase_header("Initializing Experiment")
        experiment = SelfOrganizingMemoryExperiment(
            num_memories=num_memories,
            consolidation_threshold=args.consolidation_threshold
        )
        
        print(f"Creating experiment with {num_memories} memories...")
        print(f"Consolidation threshold: {args.consolidation_threshold}")
        print()
        
        # Run the experiment
        print_phase_header("Running Experiment")
        start_time = time.time()
        
        results = experiment.run(verbose=not args.no_progress)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print results
        print_phase_header("Experiment Complete")
        print(f"Total execution time: {duration:.2f} seconds")
        print()
        
        print_results_summary(results)
        
        # Return success status
        return results.status == "PASS"
        
    except ImportError as e:
        print(f"❌ Error importing experiment module: {e}")
        print("Make sure the experiment modules are properly installed.")
        print("Expected location: fractalstat/exp08_self_organizing_memory/experiment.py")
        return False
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)