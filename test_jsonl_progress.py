#!/usr/bin/env python3
"""
Test script for JSONL progress file functionality.
"""

import os
import sys
import time
from pathlib import Path

# Test the updated progress reporter
from fractalsemantics.progress_comm import ProgressReporter, read_progress_from_file

# Set up progress file path
progress_file = Path('results/gui_progress.jsonl')
progress_file.parent.mkdir(exist_ok=True)

# Clear any existing progress file
if progress_file.exists():
    progress_file.unlink()

# Set the environment variable so the progress reporter knows where to write
os.environ['FRACTALSEMANTICS_PROGRESS_FILE'] = str(progress_file)

# Create progress reporter
progress = ProgressReporter('EXP-01')

print('Testing progress file reading functionality...')
try:
    # Test multiple progress updates
    result = progress.update(25.0, 'Stage 1', 'First update')
    print(f'First update result: {result}')
    time.sleep(0.2)

    result = progress.update(50.0, 'Stage 2', 'Second update')
    print(f'Second update result: {result}')
    time.sleep(0.2)

    result = progress.update(75.0, 'Stage 3', 'Third update')
    print(f'Third update result: {result}')
    time.sleep(0.2)

    result = progress.complete('Test completed')
    print(f'Completion result: {result}')

    # Test reading the latest progress from file
    print('\nTesting read_progress_from_file...')
    latest_progress = read_progress_from_file(progress_file)
    print(f'Latest progress: {latest_progress}')

    if latest_progress:
        print(f'Latest progress stage: {latest_progress.get("stage")}')
        print(f'Latest progress percentage: {latest_progress.get("progress")}')
        print(f'Latest progress message: {latest_progress.get("message")}')

    # Test reading all lines
    print('\nTesting reading all lines...')
    with open(progress_file) as f:
        lines = f.readlines()
        print(f'Total lines: {len(lines)}')
        for i, line in enumerate(lines):
            import json
            data = json.loads(line.strip())
            print(f'  Line {i+1}: {data["stage"]} - {data["progress"]}%')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
