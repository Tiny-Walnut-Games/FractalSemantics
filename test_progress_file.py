#!/usr/bin/env python3
"""Test script to verify progress file functionality."""

import sys

sys.path.insert(0, '.')

import json
from pathlib import Path

from fractalsemantics.progress_comm import report_progress

# Test progress file writing
progress_file = Path('results/gui_progress.jsonl')
progress_file.parent.mkdir(exist_ok=True)

print("Testing progress file writing...")

# Write some test progress messages
for i in range(0, 101, 10):
    report_progress('EXP-01', i, 'Test Stage', f'Test progress {i}%')
    print(f'Wrote progress: {i}%')

# Read back the progress
print('\nReading progress file:')
if progress_file.exists():
    with open(progress_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                print(f'Experiment: {data["experiment_id"]}, Progress: {data["progress_percent"]}%, Stage: {data["stage"]}')
else:
    print("Progress file does not exist!")
