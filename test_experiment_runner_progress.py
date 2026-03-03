"""
Test experiment runner with progress file writing.
"""

import os
import sys
from pathlib import Path

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))

from fractalsemantics.experiment_runner import ExperimentRunner


def test_experiment_runner_progress():
    """Test that experiment runner writes to progress file."""

    # Set up progress file path
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    progress_file = results_dir / "gui_progress.jsonl"
    os.environ["FRACTALSEMANTICS_PROGRESS_FILE"] = str(progress_file)

    print(f"Testing experiment runner with progress file: {progress_file}")
    print("Progress file will be monitored for updates...")

    # Create experiment runner
    runner = ExperimentRunner()

    # Run a quick experiment
    import asyncio
    result = asyncio.run(runner.run_experiment("EXP-01", quick_mode=True))

    print(f"\nExperiment result: {result.experiment_id} - Success: {result.success}")
    print(f"Duration: {result.duration:.2f}s")

    # Read progress file to see what was recorded
    print("\nProgress file contents:")
    if progress_file.exists():
        with open(progress_file) as f:
            for line in f:
                if line.strip():
                    print(line.strip())
    else:
        print("No progress file found")

if __name__ == "__main__":
    test_experiment_runner_progress()
