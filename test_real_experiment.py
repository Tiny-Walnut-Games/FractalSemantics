"""
Test script for real experiment with progress reporting.
"""

import os
import sys
import time
from pathlib import Path

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))
os.environ['FRACTALSEMANTICS_PROGRESS_FILE'] = str(Path('results/gui_progress.jsonl'))

# Import and run a simple experiment
from fractalsemantics.experiment_runner import ExperimentRunner
from fractalsemantics.progress_comm import ProgressReporter


# Create a simple test experiment
def test_experiment():
    progress = ProgressReporter('EXP-TEST')

    print('Starting test experiment...')
    progress.update(0, 'Initialization', 'Setting up experiment')
    time.sleep(0.5)

    progress.update(33, 'Data Generation', 'Generating test data')
    time.sleep(0.5)

    progress.update(66, 'Computation', 'Performing calculations')
    time.sleep(0.5)

    progress.update(100, 'Complete', 'Experiment finished successfully')

    return 'Test experiment completed'

# Run the test
result = test_experiment()
print(f'Result: {result}')

# Check progress file
progress_file = Path('results/gui_progress.jsonl')
if progress_file.exists():
    print(f'\nProgress file exists: {progress_file}')
    with open(progress_file) as f:
        lines = f.readlines()
        print(f'Number of progress entries: {len(lines)}')
        for i, line in enumerate(lines):
            import json
            data = json.loads(line.strip())
            print(f'  {i+1}. {data["stage"]} - {data["progress"]}%')
else:
    print('\nProgress file does not exist!')
