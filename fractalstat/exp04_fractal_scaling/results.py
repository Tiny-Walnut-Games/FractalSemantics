"""
EXP-04: FractalStat Fractal Scaling Test - Results Processing

This module handles saving and processing the results from the fractal scaling
test, including JSON serialization and file output.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .entities import FractalScalingResults


def save_results(
    results: FractalScalingResults, output_file: Optional[str] = None
) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp04_fractal_scaling_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path