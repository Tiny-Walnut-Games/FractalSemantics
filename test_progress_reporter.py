"""
Test ProgressReporter class directly to see if it writes to the progress file.
"""

import os
import sys
from pathlib import Path

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))

from fractalsemantics.progress_comm import (
    ProgressReporter,
    clear_progress_file,
    read_progress_from_file,
)


def test_progress_reporter():
    """Test ProgressReporter class directly."""

    # Set up progress file path
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    progress_file = results_dir / "gui_progress.jsonl"
    os.environ["FRACTALSEMANTICS_PROGRESS_FILE"] = str(progress_file)

    # Clear progress file
    clear_progress_file(progress_file)

    print(f"Testing ProgressReporter with file: {progress_file}")
    print("Progress file will be monitored for updates...")

    # Create progress reporter
    progress = ProgressReporter("TEST-EXP")

    # Send some progress messages
    print("Sending progress messages...")
    progress.update(0, "Initialization", "Starting test")
    progress.update(25, "Generation", "Generating test data")
    progress.update(50, "Processing", "Processing test data")
    progress.update(75, "Analysis", "Analyzing results")
    progress.complete("Test completed successfully")

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
    test_progress_reporter()
