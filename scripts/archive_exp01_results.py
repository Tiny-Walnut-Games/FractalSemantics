#!/usr/bin/env python3
"""
Archive EXP-01 validation results with metadata

This script runs EXP-01 and archives the results to VALIDATION_RESULTS_PHASE1.json
with complete metadata for reproducibility and publication.

Usage:
    python scripts/archive_exp01_results.py
    
Output:
    VALIDATION_RESULTS_PHASE1.json - Complete validation results with metadata
"""

import json
import platform
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def get_system_info():
    """Collect system information for reproducibility."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
    }


def get_git_info():
    """Get git commit information if available."""
    try:
        import subprocess
        
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL
        ) != 0
        
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception:
        return None


def get_dependencies():
    """Get installed package versions."""
    try:
        import pkg_resources
        
        packages = ["pydantic", "numpy", "torch", "transformers"]
        versions = {}
        
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                versions[package] = version
            except Exception:
                versions[package] = "not installed"
        
        return versions
    except Exception:
        return {}


def compute_results_checksum(results):
    """Compute SHA-256 checksum of results for verification."""
    results_json = json.dumps(results, sort_keys=True, indent=2)
    return hashlib.sha256(results_json.encode('utf-8')).hexdigest()


def main():
    """Main function to run and archive EXP-01 results."""
    print("=" * 70)
    print("EXP-01 Results Archival")
    print("=" * 70)
    print()
    
    # Import experiment (specific import for clarity and maintainability)
    try:
        from fractalstat.stat7_experiments import run_all_experiments
    except ImportError:
        print("❌ Error: Cannot import fractalstat.stat7_experiments.run_all_experiments")
        print("Make sure you're in the project root directory")
        return 1
    
    # Collect metadata
    print("Collecting system metadata...")
    metadata = {
        "experiment": "EXP-01",
        "experiment_name": "Address Uniqueness Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": get_system_info(),
        "git_info": get_git_info(),
        "dependencies": get_dependencies(),
    }
    
    print(f"  Platform: {metadata['system_info']['platform']}")
    print(f"  Python: {metadata['system_info']['python_version']}")
    if metadata['git_info']:
        print(f"  Git commit: {metadata['git_info']['commit'][:8]}")
        print(f"  Git branch: {metadata['git_info']['branch']}")
    print()
    
    # Run experiments
    print("Running EXP-01...")
    print()
    
    try:
        results = run_all_experiments(
            exp01_samples=1000,
            exp01_iterations=10,
            exp02_queries=1000,
            exp03_samples=1000
        )
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        return 1
    
    # Add metadata to results
    results["_metadata"] = metadata
    
    # Compute checksum
    checksum = compute_results_checksum(results)
    results["_metadata"]["checksum"] = checksum
    
    # Save results
    output_file = Path("VALIDATION_RESULTS_PHASE1.json")
    print()
    print(f"Saving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved successfully")
    print()
    
    # Print summary
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    exp01_summary = results.get("EXP-01", {}).get("summary", {})
    
    print(f"Total Bit-Chains Tested: {exp01_summary.get('total_bitchains_tested', 0):,}")
    print(f"Total Collisions: {exp01_summary.get('total_collisions', 0)}")
    print(f"Overall Collision Rate: {exp01_summary.get('overall_collision_rate', 0)*100:.1f}%")
    print(f"All Passed: {exp01_summary.get('all_passed', False)}")
    print()
    print(f"Checksum: {checksum}")
    print()
    
    # Verification
    print("Verifying results...")
    
    # Reload and verify checksum
    with open(output_file, 'r') as f:
        loaded_results = json.load(f)
    
    saved_checksum = loaded_results["_metadata"].pop("checksum")
    recomputed_checksum = compute_results_checksum(loaded_results)
    
    if saved_checksum == recomputed_checksum:
        print("✅ Checksum verification passed")
    else:
        print("❌ Checksum verification failed!")
        return 1
    
    print()
    print("=" * 70)
    print("✅ Archival complete!")
    print("=" * 70)
    print()
    print(f"Results file: {output_file}")
    print(f"File size: {output_file.stat().st_size:,} bytes")
    print()
    print("Next steps:")
    print("  1. Review results file")
    print("  2. Generate figures: python scripts/generate_exp01_figures.py")
    print("  3. Commit results to version control")
    print("  4. Tag release for publication")
    
    return 0


if __name__ == "__main__":
    exit(main())
