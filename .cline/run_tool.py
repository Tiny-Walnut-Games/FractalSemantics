#!/usr/bin/env python3
"""
Wrapper script for running Python tools from the markdown system.
"""
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tool.py <tool_path> [args...]")
        sys.exit(1)

    tool_path = Path(sys.argv[1])
    args = sys.argv[2:]

    if not tool_path.exists():
        print(f"Error: Tool not found: {tool_path}")
        sys.exit(1)

    # Run the tool
    result = subprocess.run([sys.executable, str(tool_path)] + args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
