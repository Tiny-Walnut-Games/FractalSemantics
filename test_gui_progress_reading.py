"""
Test script for GUI progress reading functionality.
"""

import os
import sys
from pathlib import Path

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))
os.environ['FRACTALSEMANTICS_PROGRESS_FILE'] = str(Path('results/gui_progress.jsonl'))

# Test GUI progress reading
from fractalsemantics.progress_comm import read_progress_from_file

progress_file = Path('results/gui_progress.jsonl')
print('Testing GUI progress reading...')

# Read latest progress
latest_progress = read_progress_from_file(progress_file)
print(f'Latest progress: {latest_progress}')

# Read all progress
print('\nReading all progress entries:')
with open(progress_file) as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        import json
        data = json.loads(line.strip())
        print(f'  {i+1}. {data["experiment_id"]} - {data["stage"]} - {data["progress"]}%')

print('\nGUI progress reading test completed successfully!')
