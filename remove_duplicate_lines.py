"""
remove_duplicate_lines.py — remove duplicate lines from a file while preserving order.
Usage:
    python remove_duplicate_lines.py input.txt output.txt
    python remove_duplicate_lines.py --in-place input.txt
"""

import argparse
from pathlib import Path


def remove_duplicate_lines(input_path: Path, output_path: Path):
    seen = set()
    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            if line not in seen:
                seen.add(line)
                outfile.write(line)


def dedupe_in_place(path: Path):
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    seen = set()
    unique = []

    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    with path.open("w", encoding="utf-8") as f:
        f.writelines(unique)


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate lines from a file.")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", nargs="?", help="Output file path (omit if using --in-place)")
    parser.add_argument("--in-place", action="store_true", help="Modify the file directly")

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.in_place:
        dedupe_in_place(input_path)
    else:
        if not args.output:
            raise SystemExit("Error: You must specify an output file unless using --in-place.")
        output_path = Path(args.output)
        remove_duplicate_lines(input_path, output_path)


if __name__ == "__main__":
    main()
