#!/usr/bin/env python3
"""
Fix Whitespace on Empty Lines Script

This script removes all whitespace characters from empty lines across the FractalSemantics project.
It processes all Python files and ensures clean formatting without affecting actual code content.
"""

from pathlib import Path
from typing import List, Tuple


def fix_file_whitespace(file_path: Path) -> Tuple[bool, int]:
    """
    Fix whitespace on empty lines in a single file.

    Args:
        file_path: Path to the file to fix

    Returns:
        Tuple of (was_modified, lines_changed)
    """
    try:
        # Read the file content
        with open(file_path, encoding='utf-8') as f:
            original_content = f.read()

        # Split into lines
        lines = original_content.split('\n')

        # Track changes
        modified = False
        lines_changed = 0

        # Process each line
        for i, line in enumerate(lines):
            # Check if line contains only whitespace
            if line.strip() == '' and line != '':
                # Remove all whitespace from empty lines
                lines[i] = ''
                modified = True
                lines_changed += 1

        # If file was modified, write it back
        if modified:
            new_content = '\n'.join(lines)

            # Ensure file ends with newline
            if not new_content.endswith('\n'):
                new_content += '\n'

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return True, lines_changed

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in the directory and subdirectories."""
    python_files = []

    for file_path in directory.rglob("*.py"):
        python_files.append(file_path)

    return python_files


def main():
    """Main function to fix whitespace across the project."""
    print("🔧 FractalSemantics Whitespace Fixer")
    print("=" * 50)

    # Get the project root directory
    project_root = Path(__file__).parent

    print(f"📁 Project root: {project_root}")

    # Find all Python files
    print("🔍 Finding Python files...")
    python_files = find_python_files(project_root)

    print(f"Found {len(python_files)} Python files")

    # Process each file
    total_modified = 0
    total_lines_changed = 0

    for file_path in python_files:
        was_modified, lines_changed = fix_file_whitespace(file_path)

        if was_modified:
            total_modified += 1
            total_lines_changed += lines_changed
            print(f"✅ {file_path.relative_to(project_root)}: {lines_changed} lines fixed")

    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Files processed: {len(python_files)}")
    print(f"Files modified: {total_modified}")
    print(f"Total lines fixed: {total_lines_changed}")

    if total_modified == 0:
        print("✅ No whitespace issues found - all files are clean!")
    else:
        print(f"✅ Successfully fixed whitespace in {total_modified} files")

    print("\n✨ Whitespace cleanup complete!")


if __name__ == "__main__":
    main()
