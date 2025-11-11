#!/usr/bin/env python3
"""Create a publication archive for a specific version."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def create_archive(version: str) -> None:
    """Create a complete archive of publication artifacts."""
    archive_dir = Path("publication") / "archives" / version
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating publication archive for version {version}...")

    # Copy experiment results if they exist
    results_dir = Path("results")
    if results_dir.exists():
        dest = archive_dir / "results"
        shutil.copytree(results_dir, dest, dirs_exist_ok=True)
        print(f"✓ Archived results to {dest}")

    # Copy experiment reports if they exist
    reports_dir = Path("experiment_reports")
    if reports_dir.exists():
        dest = archive_dir / "reports"
        shutil.copytree(reports_dir, dest, dirs_exist_ok=True)
        print(f"✓ Archived reports to {dest}")

    # Copy any JSON experiment outputs
    for json_file in Path(".").glob("exp*.json"):
        dest = archive_dir / "experiment_data" / json_file.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(json_file, dest)
        print(f"✓ Archived {json_file.name}")

    # Create manifest
    manifest = {
        "version": version,
        "archived_at": datetime.now().isoformat(),
        "repository": "https://gitlab.com/tiny-walnut-games/fractalstat",
        "tag": f"v{version}",
        "experiments": [
            {"id": "EXP-01", "name": "Address Uniqueness"},
            {"id": "EXP-02", "name": "Retrieval Efficiency"},
            {"id": "EXP-03", "name": "Dimension Necessity"},
            {"id": "EXP-04", "name": "Fractal Scaling"},
            {"id": "EXP-05", "name": "Compression/Expansion"},
            {"id": "EXP-06", "name": "Entanglement Detection"},
            {"id": "EXP-07", "name": "LUCA Bootstrap"},
            {"id": "EXP-08", "name": "RAG Integration"},
            {"id": "EXP-09", "name": "Concurrency"},
            {"id": "EXP-10", "name": "Bob the Skeptic"},
        ],
        "artifacts": {
            "results": "results/",
            "reports": "reports/",
            "experiment_data": "experiment_data/",
            "code_snapshot": f"https://gitlab.com/tiny-walnut-games/fractalstat/-/tree/v{version}",
        },
        "reproducibility": {
            "python_version": "3.11",
            "dependencies": "requirements.txt",
            "random_seeds": "documented in experiment code",
        },
    }

    manifest_file = archive_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Created manifest: {manifest_file}")

    # Create README for the archive
    readme = f"""# FractalStat Publication Archive - Version {version}

Archived: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Contents

- `manifest.json` - Archive metadata and experiment list
- `results/` - Raw experiment results
- `reports/` - Experiment validation reports
- `experiment_data/` - JSON data files from experiments

## Reproducibility

To reproduce these results:

1. Clone the repository at tag v{version}:
   ```bash
   git clone --branch v{version} https://gitlab.com/tiny-walnut-games/fractalstat.git
   cd fractalstat
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Run experiments:
   ```bash
   python -m fractalstat.stat7_experiments
   ```

## Citation

See `../CITATION.cff` for citation information.

## DOI

See `../doi_metadata.json` for DOI registration metadata.
"""

    readme_file = archive_dir / "README.md"
    readme_file.write_text(readme)
    print(f"✓ Created README: {readme_file}")

    print(f"\n✓ Archive complete: {archive_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: create_publication_archive.py <version>")
        print("Example: create_publication_archive.py 1.0.0")
        sys.exit(1)

    version = sys.argv[1].lstrip("v")  # Remove 'v' prefix if present
    create_archive(version)
