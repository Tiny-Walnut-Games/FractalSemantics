"""
Monitor progress file in real-time to verify progress updates are happening.
"""

import json
import time
from pathlib import Path

# Monitor progress file in real-time
progress_file = Path('results/gui_progress.jsonl')

print('Monitoring progress file...')
print('Current entries:', len(progress_file.read_text().splitlines()) if progress_file.exists() else 0)

# Watch for new entries
start_count = len(progress_file.read_text().splitlines()) if progress_file.exists() else 0

for _i in range(20):  # Watch for 10 seconds
    if progress_file.exists():
        current_count = len(progress_file.read_text().splitlines())
        if current_count > start_count:
            print(f'New entries added! Total: {current_count}')
            # Show last few entries
            lines = progress_file.read_text().splitlines()
            for line in lines[-3:]:
                data = json.loads(line)
                print(f'  {data["experiment_id"]} - {data["stage"]} - {data["progress"]}%')
            start_count = current_count
    time.sleep(0.5)
print('Monitoring complete.')
