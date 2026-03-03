"""
Test real-time progress bar animation in the GUI.
"""

import os
import sys
from pathlib import Path

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))

from fractalsemantics.experiment_runner import ExperimentRunner
from fractalsemantics.progress_comm import clear_progress_file, read_progress_from_file


async def test_real_time_progress():
    """Test real-time progress bar animation."""

    # Set up progress file path
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    progress_file = results_dir / "gui_progress.jsonl"
    os.environ["FRACTALSEMANTICS_PROGRESS_FILE"] = str(progress_file)

    # Clear progress file
    clear_progress_file(progress_file)

    print(f"Testing real-time progress with file: {progress_file}")
    print("Progress file will be monitored for updates...")

    # Create experiment runner
    runner = ExperimentRunner()

    # Run a single quick experiment to test progress updates
    try:
        print("Starting experiment with progress monitoring...")

        # Run experiment (no progress callback needed - progress is handled by subprocess)
        result = await runner.run_experiment(
            experiment_id="EXP-01",
            quick_mode=True
        )

        print(f"Experiment completed: {result.experiment_id}")
        print(f"Success: {result.success}")
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

    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    import asyncio
    asyncio.run(test_real_time_progress())

if __name__ == "__main__":
    main()
