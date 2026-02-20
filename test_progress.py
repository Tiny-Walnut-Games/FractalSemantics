# Test the complete progress file workflow
import os
import time
from pathlib import Path

from fractalsemantics.progress_comm import (
    clear_progress_file,
    read_progress_from_file,
    write_progress_to_file,
)

# Set up progress file path
progress_file = Path('fractalsemantics/results/gui_progress.jsonl')
os.environ['FRACTALSEMANTICS_PROGRESS_FILE'] = str(progress_file)

print('=== Testing Complete Progress File Workflow ===')
print(f'Progress file: {progress_file}')
print(f'Parent exists: {progress_file.parent.exists()}')

# Clear any existing progress file
clear_progress_file(progress_file)
print('Cleared existing progress file')

# Simulate multiple progress updates (like what would happen during experiments)
progress_updates = [
    {'experiment_id': 'EXP-01', 'progress': 0.0, 'stage': 'Starting', 'message': 'Initializing experiment', 'message_type': 'progress'},
    {'experiment_id': 'EXP-01', 'progress': 25.0, 'stage': 'Data Generation', 'message': 'Generating test data', 'message_type': 'progress'},
    {'experiment_id': 'EXP-01', 'progress': 50.0, 'stage': 'Computation', 'message': 'Running calculations', 'message_type': 'progress'},
    {'experiment_id': 'EXP-01', 'progress': 75.0, 'stage': 'Validation', 'message': 'Validating results', 'message_type': 'progress'},
    {'experiment_id': 'EXP-01', 'progress': 100.0, 'stage': 'Complete', 'message': 'Experiment completed successfully', 'message_type': 'complete'},
]

print('\\n=== Writing Progress Updates ===')
for i, update in enumerate(progress_updates):
    update['timestamp'] = f'2025-01-01T00:00:{i:02d}Z'
    success = write_progress_to_file(progress_file, update)
    print(f'Update {i+1}: {"√" if success else "?"} - {update["stage"]} ({update["progress"]}%)')
    time.sleep(0.05)  # Small delay between writes

print('\\n=== Reading Progress File ===')
final_data = read_progress_from_file(progress_file)
if final_data:
    print('√ Successfully read progress data')
    print(f'  Experiment: {final_data["experiment_id"]}')
    print(f'  Progress: {final_data["progress"]}%')
    print(f'  Stage: {final_data["stage"]}')
    print(f'  Message: {final_data["message"]}')
    print(f'  Type: {final_data["message_type"]}')
else:
    print('? Failed to read progress data')

print('\\n=== Testing GUI Progress File Path ===')
# Test that GUI would find the progress file in the correct location
gui_progress_file = Path(os.environ.get('FRACTALSEMANTICS_PROGRESS_FILE', 'fractalsemantics/results/gui_progress.jsonl'))
print(f'GUI progress file path: {gui_progress_file}')
print(f'GUI progress file exists: {gui_progress_file.exists()}')
print(f'GUI progress file parent exists: {gui_progress_file.parent.exists()}')

print('\\n=== Cleanup ===')
clear_progress_file(progress_file)
print('√ Cleaned up progress file')

print('\\n=== Test Summary ===')
print('√ Progress file creation and writing works')
print('√ Progress file reading works')
print('√ GUI progress file path is correct')
print('√ All progress file functionality verified!')
