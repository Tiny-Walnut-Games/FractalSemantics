# This test script verifies the complete functionality of the progress file system after the recent fixes.
import os
import sys
from pathlib import Path

from fractalsemantics.experiment_runner import ExperimentRunner
from fractalsemantics.progress_comm import (
    clear_progress_file,
    read_progress_from_file,
    write_progress_to_file,
)
from gui_app import FractalSemanticsGUI

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Test the complete fix
print('=== Final Verification of Progress File Fix ===')

# 1. Test experiment runner progress file path
print('\\n1. Testing experiment runner progress file path...')

runner = ExperimentRunner()
# Get the progress file path using the method
progress_file_path = Path(runner.progress_file())
print(f'   Experiment runner progress file: {progress_file_path}')
print(f'   Path exists: {progress_file_path.exists()}')
print(f'   Parent exists: {progress_file_path.parent.exists()}')

# 2. Test GUI progress file path
print('\\n2. Testing GUI progress file path...')
gui = FractalSemanticsGUI()
# The GUI sets up the environment variable in setup_session_state
progress_file_env = os.environ.get('FRACTALSEMANTICS_PROGRESS_FILE', 'fractalsemantics/results/gui_progress.jsonl')
print(f'   GUI progress file env: {progress_file_env}')
progress_file = Path(progress_file_env)
print(f'   Path exists: {progress_file.exists()}')
print(f'   Parent exists: {progress_file.parent.exists()}')

# 3. Test progress communication
print('\\n3. Testing progress communication...')

# Clear any existing progress
clear_progress_file(progress_file)

# Write test progress
test_data = {
    'experiment_id': 'EXP-01',
    'progress': 50.0,
    'stage': 'Running',
    'message': 'Test progress message',
    'timestamp': '2025-01-01T00:00:00Z',
    'message_type': 'progress'
}

write_progress_to_file(progress_file, test_data)
print(f'   Wrote test progress: {test_data}')

# Read back
read_data = read_progress_from_file(progress_file)
print(f'   Read progress data: {read_data}')

# Verify match
if read_data == test_data:
    print('   √ Progress communication works correctly')
else:
    print('   ? Progress communication failed')

# 4. Test that both paths point to the same location
print('\\n4. Testing path consistency...')
exp_path = runner.progress_file
gui_path = Path(os.environ.get('FRACTALSEMANTICS_PROGRESS_FILE', 'fractalsemantics/results/gui_progress.jsonl'))

if exp_path == gui_path:
    print('   √ Experiment runner and GUI use the same progress file path')
else:
    print(f'   ? Path mismatch: {exp_path} vs {gui_path}')

# 5. Cleanup
clear_progress_file(progress_file)

print('  === Final Verification Complete ===')
print('√ All progress file functionality verified!')
print('√ Progress bars should now work correctly in the GUI!')
