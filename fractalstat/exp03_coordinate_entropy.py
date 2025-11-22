"""
EXP-03: Coordinate Space Entropy Test

Quantifies the entropy contribution of each STAT7 dimension to the coordinate space,
measuring semantic disambiguation power and determining whether 7 dimensions is the
minimal necessary set.

Hypothesis:
Each dimension contributes measurable entropy to the coordinate space. Removing or
omitting a dimension reduces entropy and semantic clarity, even if collisions remain
at 0%. Dimensions with higher entropy contribution are critical for disambiguation.

Methodology:
1. Baseline: Generate N bit-chains with all 7 dimensions, measure coordinate-level entropy (pre-hash)
2. Ablation: Remove each dimension one at a time, recompute addresses, measure entropy loss
3. Analysis: Compare entropy scores and semantic disambiguation power with vs. without each dimension
4. Validation: All dimensions should show measurable entropy contribution; some may contribute disproportionately

Success Criteria:
- Baseline (all 7 dims): Entropy score approaches maximum (normalized to 1.0)
- Each dimension removal: Entropy score decreases measurably (>5% reduction)
- Semantic disambiguation power confirmed for all dimensions
- Minimal necessary set identified (≥7 dims for full expressiveness)

Statistical Significance:
- Sample size: ≥1,000 bit-chains
- Dimension combinations tested: 8 (baseline + 7 ablations)
- Entropy threshold: ≥5% reduction when dimension removed

Difference from Previous Collision-Based Test:
The previous EXP-03 (dimension_necessity.py) focused on hash collisions as the metric
for dimension necessity. This new approach measures information-theoretic entropy in
the coordinate space BEFORE hashing, providing a more nuanced understanding of how
each dimension contributes to semantic disambiguation, even when collisions are zero.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json
import sys
import numpy as np

from fractalstat.stat7_experiments import (
    generate_random_bitchain,
)


@dataclass
class EXP03_Result:
    """Results from EXP-03 coordinate entropy test."""

    dimensions_used: List[str]
    sample_size: int
    shannon_entropy: float  # Shannon entropy of coordinate space
    normalized_entropy: float  # Normalized to [0, 1]
    entropy_reduction_pct: float  # Percentage reduction from baseline
    unique_coordinates: int  # Number of unique coordinate combinations
    semantic_disambiguation_score: float  # How well dimensions separate entities
    meets_threshold: bool  # >5% entropy reduction when dimension removed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EXP03_CoordinateEntropy:
    """
    EXP-03: Coordinate Space Entropy Test

    This experiment quantifies the information-theoretic entropy contribution of
    each STAT7 dimension to the coordinate space, measuring how well each dimension
    contributes to semantic disambiguation.

    Scientific Rationale:
    The STAT7 addressing system uses 7 dimensions to create a rich semantic space.
    While the previous collision-based test showed that all dimensions are necessary
    to avoid hash collisions, this entropy-based test provides deeper insight into
    HOW MUCH each dimension contributes to the information content of the system.

    Key Concepts:
    1. **Shannon Entropy**: Measures the information content of the coordinate space.
       Higher entropy = more information = better disambiguation.

    2. **Coordinate-Level Measurement**: We measure entropy BEFORE hashing, on the
       raw coordinate space. This captures semantic structure that might be lost
       in the hash.

    3. **Ablation Testing**: By removing each dimension and measuring entropy loss,
       we quantify each dimension's contribution to the overall information content.

    4. **Semantic Disambiguation**: Beyond just avoiding collisions, we measure how
       well dimensions separate semantically different entities in the coordinate space.

    The 7 dimensions are:
    - realm: Domain classification (data, narrative, system, etc.)
    - lineage: Generation from LUCA (temporal context)
    - adjacency: Relational neighbors (graph structure)
    - horizon: Lifecycle stage (genesis, peak, decay, etc.)
    - resonance: Charge/alignment (-1.0 to 1.0)
    - velocity: Rate of change (-1.0 to 1.0)
    - density: Compression distance (0.0 to 1.0)

    By measuring entropy contribution, we can:
    - Identify which dimensions are most critical for disambiguation
    - Determine if 7 is truly the minimal necessary set
    - Understand the information structure of STAT7 space
    - Guide future dimension design decisions (e.g., STAT8?)
    """

    STAT7_DIMENSIONS = [
        "realm",
        "lineage",
        "adjacency",
        "horizon",
        "resonance",
        "velocity",
        "density",
    ]

    def __init__(self, sample_size: int = 1000, random_seed: int = 42):
        """
        Initialize the coordinate entropy experiment.

        Args:
            sample_size: Number of bit-chains to generate for testing
            random_seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.results: List[EXP03_Result] = []
        self.baseline_entropy: Optional[float] = None

    def compute_shannon_entropy(self, coordinates: List[str]) -> float:
        """
        Compute Shannon entropy of a list of coordinate representations.

        Shannon entropy H(X) = -Σ p(x) * log2(p(x))
        where p(x) is the probability of observing coordinate value x.

        Higher entropy indicates more information content and better
        discrimination between different entities.

        Args:
            coordinates: List of coordinate string representations

        Returns:
            Shannon entropy in bits
        """
        if not coordinates:
            return 0.0

        # Count frequency of each unique coordinate
        counts = Counter(coordinates)
        total = len(coordinates)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return entropy

    def normalize_entropy(self, entropy: float, num_samples: int) -> float:
        """
        Normalize entropy to [0, 1] range.

        Maximum possible entropy occurs when all coordinates are unique,
        which gives H_max = log2(num_samples).

        Normalized entropy = H / H_max

        Args:
            entropy: Raw Shannon entropy
            num_samples: Number of samples (for computing max entropy)

        Returns:
            Normalized entropy in [0, 1]
        """
        if num_samples <= 1:
            return 0.0

        max_entropy = np.log2(num_samples)
        if max_entropy == 0:
            return 0.0

        return min(1.0, entropy / max_entropy)

    def compute_semantic_disambiguation_score(
        self, coordinates: List[str], num_unique: int
    ) -> float:
        """
        Compute semantic disambiguation score.

        This metric measures how well the coordinate space separates
        semantically different entities. It combines:
        1. Uniqueness ratio (unique coordinates / total samples)
        2. Distribution uniformity (how evenly spread the coordinates are)

        A high score indicates good semantic separation.

        Args:
            coordinates: List of coordinate string representations
            num_unique: Number of unique coordinates

        Returns:
            Disambiguation score in [0, 1]
        """
        if not coordinates:
            return 0.0

        # Uniqueness ratio
        uniqueness = num_unique / len(coordinates)

        # Distribution uniformity (using entropy as proxy)
        counts = Counter(coordinates)
        total = len(coordinates)

        # Compute normalized entropy of distribution
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        uniformity = entropy / max_entropy if max_entropy > 0 else 0.0

        # Combine metrics (weighted average)
        score = 0.7 * uniqueness + 0.3 * uniformity

        return score

    def extract_coordinates(self, bitchains: List, dimensions: List[str]) -> List[str]:
        """
        Extract coordinate representations from bit-chains using specified dimensions.

        Creates a canonical string representation of each bit-chain's coordinates
        using only the specified dimensions. This allows us to measure entropy
        with different dimension subsets.

        Args:
            bitchains: List of BitChain objects
            dimensions: List of dimension names to include

        Returns:
            List of coordinate string representations
        """
        coordinates = []

        for bc in bitchains:
            # Get canonical dict
            data = bc.to_canonical_dict()
            coords = data["stat7_coordinates"]

            # Filter to only specified dimensions
            filtered_coords = {k: v for k, v in coords.items() if k in dimensions}

            # Create canonical string representation
            # Sort keys for consistency
            coord_str = json.dumps(filtered_coords, sort_keys=True)
            coordinates.append(coord_str)

        return coordinates

    def run(self) -> Tuple[List[EXP03_Result], bool]:
        """
        Run the coordinate entropy test.

        Returns:
            Tuple of (results list, overall success boolean)
        """
        print(f"\n{'=' * 70}")
        print("EXP-03: COORDINATE SPACE ENTROPY TEST")
        print(f"{'=' * 70}")
        print(f"Sample size: {self.sample_size} bit-chains")
        print(f"Random seed: {self.random_seed} (for reproducibility)")
        print()

        # Generate bit-chains once for all tests
        print("Generating bit-chains...")
        bitchains = [
            generate_random_bitchain(seed=self.random_seed + i)
            for i in range(self.sample_size)
        ]
        print(f"Generated {len(bitchains)} bit-chains")
        print()

        # Baseline: all 7 dimensions
        print("=" * 70)
        print("BASELINE: All 7 dimensions")
        print("=" * 70)

        baseline_coords = self.extract_coordinates(bitchains, self.STAT7_DIMENSIONS)
        baseline_unique = len(set(baseline_coords))
        baseline_entropy = self.compute_shannon_entropy(baseline_coords)
        baseline_normalized = self.normalize_entropy(baseline_entropy, self.sample_size)
        baseline_disambiguation = self.compute_semantic_disambiguation_score(
            baseline_coords, baseline_unique
        )

        self.baseline_entropy = baseline_entropy

        result = EXP03_Result(
            dimensions_used=self.STAT7_DIMENSIONS.copy(),
            sample_size=self.sample_size,
            shannon_entropy=baseline_entropy,
            normalized_entropy=baseline_normalized,
            entropy_reduction_pct=0.0,  # Baseline has no reduction
            unique_coordinates=baseline_unique,
            semantic_disambiguation_score=baseline_disambiguation,
            meets_threshold=True,  # Baseline always meets threshold
        )
        self.results.append(result)

        print(f"  Shannon Entropy:      {baseline_entropy:.4f} bits")
        print(f"  Normalized Entropy:   {baseline_normalized:.4f}")
        print(f"  Unique Coordinates:   {baseline_unique} / {self.sample_size}")
        print(f"  Disambiguation Score: {baseline_disambiguation:.4f}")
        print()

        # Ablation: remove each dimension
        all_success = True

        for removed_dim in self.STAT7_DIMENSIONS:
            print("=" * 70)
            print(f"ABLATION: Remove '{removed_dim}'")
            print("=" * 70)

            # Get dimensions without the removed one
            remaining_dims = [d for d in self.STAT7_DIMENSIONS if d != removed_dim]

            # Extract coordinates without this dimension
            ablation_coords = self.extract_coordinates(bitchains, remaining_dims)
            ablation_unique = len(set(ablation_coords))
            ablation_entropy = self.compute_shannon_entropy(ablation_coords)
            ablation_normalized = self.normalize_entropy(
                ablation_entropy, self.sample_size
            )
            ablation_disambiguation = self.compute_semantic_disambiguation_score(
                ablation_coords, ablation_unique
            )

            # Calculate entropy reduction
            entropy_reduction_pct = (
                (baseline_entropy - ablation_entropy) / baseline_entropy * 100
                if baseline_entropy > 0
                else 0.0
            )

            # Check if meets threshold (>5% reduction)
            meets_threshold = bool(entropy_reduction_pct > 5.0)

            result = EXP03_Result(
                dimensions_used=remaining_dims,
                sample_size=self.sample_size,
                shannon_entropy=ablation_entropy,
                normalized_entropy=ablation_normalized,
                entropy_reduction_pct=entropy_reduction_pct,
                unique_coordinates=ablation_unique,
                semantic_disambiguation_score=ablation_disambiguation,
                meets_threshold=meets_threshold,
            )
            self.results.append(result)

            # Determine if dimension is critical
            status = "[CRITICAL]" if meets_threshold else "[OPTIONAL]"
            print(f"  {status}")
            print(f"  Shannon Entropy:      {ablation_entropy:.4f} bits")
            print(f"  Normalized Entropy:   {ablation_normalized:.4f}")
            print(f"  Entropy Reduction:    {entropy_reduction_pct:.2f}%")
            print(f"  Unique Coordinates:   {ablation_unique} / {self.sample_size}")
            print(f"  Disambiguation Score: {ablation_disambiguation:.4f}")
            print()

            # For overall success, we expect all dimensions to be critical
            all_success = all_success and meets_threshold

        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Actually, we want to identify which dimensions ARE critical
        # A dimension is critical if removing it causes >5% entropy reduction
        critical_count = sum(1 for r in self.results[1:] if r.meets_threshold)

        print(f"Critical dimensions (>5% entropy reduction): {critical_count} / 7")
        print()

        # Rank dimensions by entropy contribution
        print("Dimension Ranking by Entropy Contribution:")
        ranked = sorted(
            self.results[1:],  # Skip baseline
            key=lambda r: r.entropy_reduction_pct,
            reverse=True,
        )

        for i, result in enumerate(ranked, 1):
            removed_dim = [
                d for d in self.STAT7_DIMENSIONS if d not in result.dimensions_used
            ][0]
            status = "✓" if result.meets_threshold else "✗"
            print(
                f"  {i}. {removed_dim:12s} - {
                    result.entropy_reduction_pct:6.2f}% reduction {status}"
            )

        print()

        if all_success:
            print(
                "✅ RESULT: All 7 dimensions are critical for semantic disambiguation"
            )
            print("   (all show >5% entropy reduction when removed)")
        else:
            print("⚠️  RESULT: Some dimensions may be optional")
            print("   (not all show >5% entropy reduction when removed)")

        return self.results, bool(all_success)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "baseline_entropy": self.baseline_entropy,
            "total_tests": len(self.results),
            "total_dimension_combos_tested": len(self.results),
            "all_critical": all(
                r.meets_threshold for r in self.results[1:]
            ),  # Skip baseline
            "results": [r.to_dict() for r in self.results],
        }

    def generate_visualization_data(self) -> Dict[str, Any]:
        """
        Generate data for entropy contribution visualization.

        Returns:
            Dictionary with visualization data for plotting
        """
        if len(self.results) < 2:
            return {}

        # Extract dimension names and entropy reductions
        dimensions = []
        entropy_reductions = []

        for result in self.results[1:]:  # Skip baseline
            removed_dim = [
                d for d in self.STAT7_DIMENSIONS if d not in result.dimensions_used
            ][0]
            dimensions.append(removed_dim)
            entropy_reductions.append(result.entropy_reduction_pct)

        return {
            "dimensions": dimensions,
            "entropy_reductions": entropy_reductions,
            "baseline_entropy": self.baseline_entropy,
            "threshold": 5.0,
        }


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp03_coordinate_entropy_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="UTF-8") as f:
        json.dump(results, f, indent=2)
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
        print("⚠️  matplotlib not available - skipping visualization")
        return

    if not viz_data:
        print("⚠️  No visualization data available")
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
    ax.set_xlabel("STAT7 Dimension", fontsize=12, fontweight="bold")
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
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"exp03_entropy_chart_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    # Load from config or use defaults
    try:
        from fractalstat.config import ExperimentConfig

        config = ExperimentConfig()
        sample_size = config.get("EXP-03", "sample_size", 1000)
        random_seed = config.get("EXP-03", "random_seed", 42)
    except Exception:
        sample_size = 1000
        random_seed = 42

        if "--quick" in sys.argv:
            sample_size = 100
        elif "--full" in sys.argv:
            sample_size = 5000

    try:
        experiment = EXP03_CoordinateEntropy(
            sample_size=sample_size, random_seed=random_seed
        )
        results_list, success = experiment.run()
        summary = experiment.get_summary()

        output_file = save_results(summary)

        # Generate visualization
        viz_data = experiment.generate_visualization_data()
        if viz_data:
            plot_entropy_contributions(viz_data)

        print("\n" + "=" * 70)
        print("[OK] EXP-03 COMPLETE")
        print("=" * 70)
        print(f"Results: {output_file}")
        print()

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
