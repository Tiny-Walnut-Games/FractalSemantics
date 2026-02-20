#!/usr/bin/env python3
"""
Test script for complete progress bar fix with real experiment.
"""

import os
import sys
import time
from pathlib import Path

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))
os.environ['FRACTALSEMANTICS_PROGRESS_FILE'] = str(Path('results/gui_progress.jsonl'))

# Import experiment runner
from fractalsemantics.experiment_runner import ExperimentRunner

# Create runner and run a simple experiment
runner = ExperimentRunner()

print('Testing real experiment with progress...')
print('Running EXP-01 (Geometric Collision)...')

# Run a single experiment with progress tracking
try:
    import asyncio
    result = asyncio.run(runner.run_experiment('EXP-01', quick_mode=True))
    print(f'Experiment completed: {result.success}')
    print(f'Duration: {result.duration:.2f}s')
    print(f'Result type: {result.result_type}')

    # Check progress file
    progress_file = Path('results/gui_progress.jsonl')
    if progress_file.exists():
        with open(progress_file) as f:
            lines = f.readlines()
            print(f'Progress entries recorded: {len(lines)}')

            # Show last few progress updates
            for i, line in enumerate(lines[-5:]):
                import json
                data = json.loads(line.strip())
                print(f'  {len(lines)-5+i+1}. {data["experiment_id"]} - {data["stage"]} - {data["progress"]}%')
    else:
        print('No progress file found!')

except Exception as e:
    print(f'Error running experiment: {e}')
    import traceback
    traceback.print_exc()
