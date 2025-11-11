#!/usr/bin/env python3
"""Generate publication-ready metadata for DOI minting."""

import json
import sys
from datetime import datetime
from pathlib import Path


def generate_doi_metadata(version: str) -> dict:
    """Generate metadata for DOI registration (Zenodo/Figshare format)."""
    metadata = {
        "title": "FractalStat: A 7-Dimensional Addressing System for Fractal Information Spaces",
        "version": version,
        "upload_type": "software",
        "publication_date": datetime.now().strftime("%Y-%m-%d"),
        "creators": [
            {
                "name": "[Your Name]",
                "affiliation": "Tiny Walnut Games",
                "orcid": "[Your ORCID if available]",
            }
        ],
        "description": (
            "FractalStat is a research package containing 10 validation experiments "
            "that prove the STAT7 addressing system works at scale. STAT7 is a "
            "7-dimensional coordinate system for uniquely addressing data in fractal "
            "information spaces. The 7 dimensions are: Realm, Lineage, Adjacency, "
            "Horizon, Resonance, Velocity, and Density."
        ),
        "access_right": "open",
        "license": "MIT",
        "keywords": [
            "fractal",
            "addressing-system",
            "7-dimensional",
            "information-retrieval",
            "semantic-search",
            "STAT7",
            "RAG",
            "vector-database",
        ],
        "related_identifiers": [
            {
                "identifier": "https://gitlab.com/tiny-walnut-games/fractalstat",
                "relation": "isSupplementTo",
                "resource_type": "software",
            }
        ],
        "contributors": [],
        "references": [],
        "notes": (
            "This software package includes 10 comprehensive validation experiments "
            "demonstrating the effectiveness of the STAT7 7-dimensional addressing system."
        ),
    }
    return metadata


def generate_citation_cff(version: str) -> str:
    """Generate CITATION.cff content."""
    cff = f"""cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "FractalStat: A 7-Dimensional Addressing System for Fractal Information Spaces"
version: {version}
date-released: {datetime.now().strftime("%Y-%m-%d")}
authors:
  - family-names: "[Your Last Name]"
    given-names: "[Your First Name]"
    affiliation: "Tiny Walnut Games"
    orcid: "[Your ORCID if available]"
repository-code: "https://gitlab.com/tiny-walnut-games/fractalstat"
license: MIT
keywords:
  - fractal
  - addressing-system
  - 7-dimensional
  - information-retrieval
  - semantic-search
  - STAT7
"""
    return cff


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_doi_metadata.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    pub_dir = Path("publication")
    pub_dir.mkdir(exist_ok=True)

    # Generate DOI metadata
    doi_metadata = generate_doi_metadata(version)
    doi_file = pub_dir / "doi_metadata.json"
    doi_file.write_text(json.dumps(doi_metadata, indent=2))
    print(f"✓ Generated DOI metadata: {doi_file}")

    # Generate CITATION.cff
    citation_cff = generate_citation_cff(version)
    citation_file = pub_dir / "CITATION.cff"
    citation_file.write_text(citation_cff)
    print(f"✓ Generated CITATION.cff: {citation_file}")

    # Write version file
    version_file = pub_dir / "VERSION"
    version_file.write_text(f"{version}\n")
    print(f"✓ Generated VERSION file: {version_file}")


if __name__ == "__main__":
    main()
