"""
Test script to verify progress file updates and animation.
"""

import json
import subprocess
import time
from pathlib import Path


def test_progress_animation():
    """Test progress file updates during experiment execution."""

    # Clear progress file first
    progress_file = Path('fractalsemantics/results/gui_progress.jsonl')
    if progress_file.exists():
        progress_file.unlink()

    print('Starting experiment with progress monitoring...')

    # Run a quick experiment
    proc = subprocess.Popen([
        'python', '-c', '''
import sys
sys.path.insert(0, ".")
from fractalsemantics.experiment_runner import ExperimentRunner
runner = ExperimentRunner()
runner.run_experiment("EXP-01", quick_mode=True)
'''
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Monitor progress file while experiment runs
    start_time = time.time()
    last_count = 0

    while proc.poll() is None and time.time() - start_time < 60:
        if progress_file.exists():
            lines = progress_file.read_text().splitlines()
            if lines and len(lines) > last_count:
                print(f'Progress entries: {len(lines)}')
                # Show last entry
                try:
                    data = json.loads(lines[-1])
                    print(f'  {data["experiment_id"]} - {data["stage"]} - {data["progress"]}%')
                    last_count = len(lines)
                except Exception as e:
                    print(f'  Error parsing progress: {e}')
        time.sleep(0.5)

    # Wait for completion
    proc.wait()
    print('Experiment completed.')

    if progress_file.exists():
        lines = progress_file.read_text().splitlines()
        print(f'Final progress entries: {len(lines)}')

        # Show last few entries
        for line in lines[-5:]:
            try:
                data = json.loads(line)
                print(f'  {data["experiment_id"]} - {data["stage"]} - {data["progress"]}%')
            except Exception as e:
                print(f'  Error parsing progress: {e}')
                pass
    else:
        print('No progress file created.')

if __name__ == "__main__":
    test_progress_animation()
