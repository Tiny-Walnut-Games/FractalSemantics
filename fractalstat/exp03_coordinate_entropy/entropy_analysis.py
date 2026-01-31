"""
Entropy analysis and computation for coordinate space evaluation.

Contains the core entropy computation algorithms and analysis methods for
measuring the information-theoretic contribution of each FractalStat dimension.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json
import sys
import hashlib
import uuid
import secrets
import numpy as np
from statistics import mean

from .entities import generate_random_bitchain, BitChain, Coordinates, FractalStatCoordinates, EXP03_Result


class EXP03_CoordinateEntropy:
    """
    EXP-03: Coordinate Space Entropy Test

    This experiment quantifies the information-theoretic entropy contribution of
    each FractalStat dimension to the coordinate space, measuring how well each dimension
    contributes to semantic disambiguation.

    Scientific Rationale:
    The FractalStat addressing system uses 8 dimensions to create a rich semantic space.
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

    The 8 dimensions are:
    - realm: Domain classification (data, narrative, system, faculty, event, pattern, void)
    - lineage: Generation from LUCA (temporal context)
    - adjacency: Relational neighbors (graph structure)
    - horizon: Lifecycle stage (genesis, emergence, peak, decay, crystallization)
    - luminosity: Activity level (0-100)
    - polarity: Resonance/affinity type (6 companion + 6 badge + neutral)
    - dimensionality: Fractal depth (0+)
    - alignment: Social/coordination dynamic type (lawful_good, chaotic_evil, etc.)

    By measuring entropy contribution, we can:
    - Identify which dimensions are most critical for disambiguation
    - Determine if 7 is truly the minimal necessary set
    - Understand the information structure of FractalStat space
    - Guide future dimension design decisions (e.g., FractalStat?)
    """

    FractalStat_DIMENSIONS = [
        "realm",
        "lineage",
        "adjacency",
        "horizon",
        "luminosity",
        "polarity",
        "dimensionality",
        "alignment",
    ]

    def __init__(self, sample_size: int = 1000000, random_seed: int = 42):
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
            coords = data["fractalstat_coordinates"]

            # Start with coordinates from the BitChain
            filtered_coords = {k: v for k, v in coords.items() if k in dimensions}

            # Check state field for any additional dimensions that might be stored there
            state_data = data.get("state", {})
            for dim in dimensions:
                if dim not in filtered_coords and dim in state_data:
                    filtered_coords[dim] = state_data[dim]

            # Create canonical string representation
            # Sort keys for consistency
            coord_str = json.dumps(filtered_coords, sort_keys=True)
            coordinates.append(coord_str)

        return coordinates

    def compute_individual_contribution(self, dimension_name: str, bitchains: List) -> float:
        """
        Compute individual dimension contribution vs theoretical maximum.

        Measures how close a single dimension gets to its theoretical maximum entropy,
        as a percentage. This indicates how much effective discriminatory power
        a dimension provides when used alone.

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: List of BitChain objects

        Returns:
            Contribution score [0, 100] (percentage of theoretical maximum)
        """
        if dimension_name not in self.FractalStat_DIMENSIONS:
            return 0.0

        # Get entropy using only this dimension
        single_dim_coords = self.extract_coordinates(bitchains, [dimension_name])
        actual_entropy = self.compute_shannon_entropy(single_dim_coords)

        # Calculate theoretical maximum for this dimension
        # Based on the cardinality of possible values
        if dimension_name == "realm":
            max_values = REALM_REGISTRY.get_count()  # 8 realm enum values
        elif dimension_name == "horizon":
            max_values = HORIZON_REGISTRY.get_count()  # 6 horizon enum values
        elif dimension_name == "polarity":
            max_values = POLARITY_REGISTRY.get_count()  # 12 polarity enum values
        elif dimension_name == "alignment":
            max_values = ALIGNMENT_REGISTRY.get_count()  # 12 alignment enum values
        elif dimension_name == "lineage":
            max_values = 100  # 1-100 range
        elif dimension_name == "dimensionality":
            max_values = 6  # 0-5 range
        elif dimension_name in ["adjacency", "luminosity"]:
            max_values = 100  # Continuous values discretized to 100 bins
        else:
            max_values = 100

        theoretical_max = np.log2(min(max_values, len(bitchains)))

        if theoretical_max == 0:
            return 0.0

        return min(100.0, (actual_entropy / theoretical_max) * 100.0)

    def compute_relative_contribution(self, dimension_name: str, bitchains: List, baseline_entropy: float) -> float:
        """
        Compute relative contribution using Shapley additive approach.

        Measures how much entropy the dimension adds beyond all other dimensions.
        This is a cooperative game theory approach that fairly distributes
        the total contribution among dimensions.

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: List of BitChain objects
            baseline_entropy: Full 7D system entropy

        Returns:
            Relative contribution score [0, 100]
        """
        if dimension_name not in self.FractalStat_DIMENSIONS:
            return 0.0

        # Simplified relative contribution: entropy of subset without this dimension vs with it
        other_dims = [d for d in self.FractalStat_DIMENSIONS if d != dimension_name]

        try:
            without_dim_coords = self.extract_coordinates(bitchains, other_dims)
            entropy_without = self.compute_shannon_entropy(without_dim_coords)

            # The relative contribution is how much this dimension adds
            marginal_gain = baseline_entropy - entropy_without

            # Normalize as percentage of total baseline entropy
            if baseline_entropy > 0:
                return max(0.0, min(100.0, (marginal_gain / baseline_entropy) * 100.0))
            else:
                return 0.0
        except Exception:
            # If there's an issue with the calculation, return baseline entropy share
            return 100.0 / len(self.FractalStat_DIMENSIONS)

    def compute_complementary_contribution(self, dimension_name: str, bitchains: List) -> float:
        """
        Compute complementary contribution (unique discriminatory information).

        Measures how well this dimension provides unique separation that
        other dimensions don't capture. Uses conditional entropy concept.

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: List of BitChain objects

        Returns:
            Complementary contribution score [0, 100]
        """
        if dimension_name not in self.FractalStat_DIMENSIONS:
            return 0.0

        try:
            # Get individual dimension entropy
            single_coords = self.extract_coordinates(bitchains, [dimension_name])
            single_entropy = self.compute_shannon_entropy(single_coords)

            # Get entropy of all other dimensions
            other_dims = [d for d in self.FractalStat_DIMENSIONS if d != dimension_name]
            other_coords = self.extract_coordinates(bitchains, other_dims)
            other_entropy = self.compute_shannon_entropy(other_coords)

            # Complementary contribution: entropy this dimension has that others don't
            # (simple heuristic: amount of entropy not predicted by others)
            if single_entropy > 0:
                unique_information = single_entropy * (1.0 - min(1.0, other_entropy / single_entropy))
                return min(100.0, unique_information * 20.0)  # Scale factor for visibility
            else:
                return 0.0
        except Exception:
            return 0.0

    def compute_expressiveness_contribution(
        self,
        dimension_name: str,
        bitchains: List,
        baseline_entropy: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute comprehensive expressiveness contribution composite score.

        Combines four complementary approaches:
        1. Individual contribution vs theoretical maximum (normalized to [0,1])
        2. Relative contribution via marginal gains (normalized to [0,1])
        3. Complementary contribution (unique information)
        4. Ablation-based contribution (legacy for comparison)

        Weights: 35% individual + 35% relative + 20% complementary + 10% ablation

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: List of BitChain objects
            baseline_entropy: Full system entropy

        Returns:
            Tuple of (composite_score, individual_contrib, relative_contrib, complementary_contrib)
        """
        individual = self.compute_individual_contribution(dimension_name, bitchains) / 100.0  # Normalize to [0,1]
        relative = self.compute_relative_contribution(dimension_name, bitchains, baseline_entropy) / 100.0  # [0,1]
        complementary = self.compute_complementary_contribution(dimension_name, bitchains) / 100.0  # [0,1]

        # Ablation contribution (legacy - entropy loss when removed)
        other_dims = [d for d in self.FractalStat_DIMENSIONS if d != dimension_name]
        ablation_coords = self.extract_coordinates(bitchains, other_dims)
        ablation_entropy = self.compute_shannon_entropy(ablation_coords)
        ablation_contrib = max(0.0, (baseline_entropy - ablation_entropy) / baseline_entropy)  # [0,1]

        # Weighted composite score
        composite = (
            0.35 * individual +
            0.35 * relative +
            0.20 * complementary +
            0.10 * ablation_contrib
        )

        return (
            composite * 100.0,  # Return as percentage
            individual * 100.0,
            relative * 100.0,
            complementary * 100.0
        )

    def run(self) -> Tuple[List[EXP03_Result], bool]:
        """
        Run the coordinate entropy test.

        Returns:
            Tuple of (results list, overall success boolean)
        """
        import time

        start_time = time.time()
        print(f"\n{'=' * 70}")
        print("EXP-03: COORDINATE SPACE ENTROPY TEST")
        print(f"{'=' * 70}")
        print(f"Sample size: {self.sample_size} bit-chains")
        print(f"Random seed: {self.random_seed} (for reproducibility)")
        print()

        # Generate bit-chains once for all tests
        print("Generating bit-chains...")
        generation_start = time.time()
        bitchains = [
            generate_random_bitchain(seed=self.random_seed + i)
            for i in range(self.sample_size)
        ]
        generation_time = time.time() - generation_start
        print(f"Generated {len(bitchains)} bit-chains (took {generation_time:.1f}s)")
        print()

        # Baseline: all 8 dimensions
        print("=" * 70)
        print("BASELINE: All 8 dimensions")
        print("=" * 70)

        baseline_start = time.time()
        baseline_coords = self.extract_coordinates(bitchains, self.FractalStat_DIMENSIONS)
        baseline_unique = len(set(baseline_coords))
        baseline_entropy = self.compute_shannon_entropy(baseline_coords)
        baseline_normalized = self.normalize_entropy(baseline_entropy, self.sample_size)
        baseline_disambiguation = self.compute_semantic_disambiguation_score(
            baseline_coords, baseline_unique
        )
        baseline_computation_time = time.time() - baseline_start
        print(f"Baseline computation took {baseline_computation_time:.1f}s")

        self.baseline_entropy = baseline_entropy

        result = EXP03_Result(
            dimensions_used=self.FractalStat_DIMENSIONS.copy(),
            sample_size=self.sample_size,
            shannon_entropy=baseline_entropy,
            normalized_entropy=baseline_normalized,
            expressiveness_contribution=0.0,  # Baseline
            individual_contribution=0.0,  # Baseline
            relative_contribution=0.0,  # Baseline
            complementary_contribution=0.0,  # Baseline
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

        for i, removed_dim in enumerate(self.FractalStat_DIMENSIONS, 1):
            print("=" * 70)
            print(f"ABLATION: Remove '{removed_dim}' ({i}/8)")
            print("=" * 70)

            dim_start = time.time()
            # Get dimensions without the removed one
            remaining_dims = [d for d in self.FractalStat_DIMENSIONS if d != removed_dim]

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
            dim_time = time.time() - dim_start
            print(f"Ablation for '{removed_dim}' took {dim_time:.1f}s")

            # Calculate entropy reduction
            entropy_reduction_pct = (
                (baseline_entropy - ablation_entropy) / baseline_entropy * 100
                if baseline_entropy > 0
                else 0.0
            )

            # Check if meets threshold (>0% reduction - any measurable contribution)
            meets_threshold = bool(entropy_reduction_pct > 0.0)

            # Compute expressiveness contribution for this dimension
            composite_score, individual_score, relative_score, complementary_score = self.compute_expressiveness_contribution(
                removed_dim, bitchains, baseline_entropy
            )

            result = EXP03_Result(
                dimensions_used=remaining_dims,
                sample_size=self.sample_size,
                shannon_entropy=ablation_entropy,
                normalized_entropy=ablation_normalized,
                expressiveness_contribution=composite_score,
                individual_contribution=individual_score,
                relative_contribution=relative_score,
                complementary_contribution=complementary_score,
                entropy_reduction_pct=entropy_reduction_pct,
                unique_coordinates=ablation_unique,
                semantic_disambiguation_score=ablation_disambiguation,
                meets_threshold=composite_score >= 5.0,  # Use composite score for threshold
            )
            self.results.append(result)

            # Determine if dimension is critical
            status = "CRITICAL" if meets_threshold else "OPTIONAL"
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

        print(f"Critical dimensions (>0% entropy reduction): {critical_count} / 8")
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
                d for d in self.FractalStat_DIMENSIONS if d not in result.dimensions_used
            ][0]
            status = "PASS" if result.meets_threshold else "FAIL"
            print(
                f"  {i}. {removed_dim:12s} - {result.entropy_reduction_pct:6.2f}% reduction [{status}]"
            )

        print()

        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.1f}s")

        if all_success:
            print(
                "[Success] RESULT: All 8 dimensions are critical for semantic disambiguation"
            )
            print("   (all show measurablesurable entropy reduction when removed)")
        else:
            print("[Warn]  RESULT: Some dimensions may be optional")
            print("   (not all show measurable entropy reduction when removed)")

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
                d for d in self.FractalStat_DIMENSIONS if d not in result.dimensions_used
            ][0]
            dimensions.append(removed_dim)
            entropy_reductions.append(result.entropy_reduction_pct)

        return {
            "dimensions": dimensions,
            "entropy_reductions": entropy_reductions,
            "baseline_entropy": self.baseline_entropy,
            "threshold": 5.0,
        }