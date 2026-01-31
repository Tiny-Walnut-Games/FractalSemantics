"""
Visualization utilities for coordinate entropy analysis.

Contains functions for generating charts and plots to visualize entropy
contribution data and dimension analysis results.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp03_coordinate_entropy_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    # Create JSON-serializable copy with robust type handling
    def make_serializable(obj):
        # Handle boolean types first (including numpy bools)
        if hasattr(obj, 'dtype'):  # numpy array/scalar
            if obj.dtype == 'bool':
                return bool(obj)
            elif obj.dtype.kind in ['i', 'u']:  # integer types
                return int(obj)
            elif obj.dtype.kind in ['f', 'c']:  # float/complex types
                return float(obj)
            else:
                return str(obj)  # fallback for other numpy types
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [make_serializable(item) for item in obj]  # convert tuples to lists
        else:
            # For any other object types, convert to string as fallback
            return str(obj)

    serializable_results = make_serializable(results)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(serializable_results, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


def plot_entropy_contributions(
    viz_data: Dict[str, Any], output_file: Optional[str] = None
):
    """
    Generate entropy contribution visualization.

    Creates a bar chart showing entropy reduction when each dimension is removed.

    Args:
        viz_data: Visualization data from generate_visualization_data()
        output_file: Optional output file path for saving the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn]  matplotlib not available - skipping visualization")
        return

    if not viz_data:
        print("[Warn]  No visualization data available")
        return

    dimensions = viz_data["dimensions"]
    entropy_reductions = viz_data["entropy_reductions"]
    threshold = viz_data["threshold"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    colors = ["green" if er > threshold else "orange" for er in entropy_reductions]
    bars = ax.bar(dimensions, entropy_reductions, color=colors, alpha=0.7)

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold}%)",
    )

    # Labels and title
    ax.set_xlabel("FractalStat Dimension", fontsize=12, fontweight="bold")
    ax.set_ylabel("Entropy Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "EXP-03: Entropy Contribution by Dimension\n(When Dimension is Removed)",
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels on bars
    for bar, value in zip(bars, entropy_reductions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Legend
    ax.legend()

    # Grid
    ax.grid(axis="y", alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {output_file}")
    else:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"exp03_entropy_chart_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.close()