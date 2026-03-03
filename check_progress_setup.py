"""
Check progress file setup and environment.
"""

import os
from pathlib import Path


def check_progress_setup():
    """Check progress file setup and environment."""

    # Check progress file path
    progress_file = Path('results/gui_progress.jsonl')
    print(f'Progress file path: {progress_file.absolute()}')
    print(f'Progress file exists: {progress_file.exists()}')

    # Check environment variable
    env_var = os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE", "Not set")
    print(f'Environment variable: {env_var}')

    # Check if results directory exists
    results_dir = Path('results')
    print(f'Results directory exists: {results_dir.exists()}')

    if results_dir.exists():
        contents = list(results_dir.iterdir())
        print(f'Results directory contents: {contents}')
    else:
        print('Results directory does not exist.')

    # Check if we can create the file
    try:
        results_dir.mkdir(exist_ok=True)
        test_file = results_dir / 'test_progress.jsonl'
        test_file.write_text('{"test": "data"}\n')
        print(f'Can create files in results directory: {test_file.exists()}')
        test_file.unlink()
    except Exception as e:
        print(f'Cannot create files in results directory: {e}')

if __name__ == "__main__":
    check_progress_setup()
