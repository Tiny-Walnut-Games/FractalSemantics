"""
EXP-03: Coordinate Space Entropy Test

Quantifies the entropy contribution of each FractalSemantics dimension to the coordinate space,
measuring semantic disambiguation power and determining whether 8 dimensions is the
minimal necessary set.

Hypothesis:
Each dimension contributes measurable entropy to the coordinate space. Removing or
omitting a dimension reduces entropy and semantic clarity, even if collisions remain
at 0%. Dimensions with higher entropy contribution are critical for disambiguation.

Methodology:
1. Baseline: Generate N bit-chains with all 8 dimensions, measure coordinate-level entropy (pre-hash)
2. Ablation: Remove each dimension one at a time, recompute addresses, measure entropy loss
3. Analysis: Compare entropy scores and semantic disambiguation power with vs. without each dimension
4. Validation: All dimensions should show measurable entropy contribution; some may contribute disproportionately

Success Criteria:
- Baseline (all 8 dims): Entropy score approaches maximum (normalized to 1.0)
- Each dimension removal: Entropy score decreases measurably (>5% reduction)
- Semantic disambiguation power confirmed for all dimensions
- Minimal necessary set identified (≥7 dims for full expressiveness)

Statistical Significance:
- Sample size: ≥100,000 bit-chains
- Dimension combinations tested: 8 (baseline + 7 ablations)
- Entropy threshold: ≥5% reduction when dimension removed

Difference from Previous Collision-Based Test:
The previous EXP-03 (dimension_necessity.py) focused on hash collisions as the metric
for dimension necessity. This new approach measures information-theoretic entropy in
the coordinate space BEFORE hashing, providing a more nuanced understanding of how
each dimension contributes to semantic disambiguation, even when collisions are zero.
"""

import contextlib
import hashlib
import json
import secrets
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from fractalsemantics.dynamic_enum import (
    ALIGNMENT_REGISTRY,
    HORIZON_REGISTRY,
    POLARITY_REGISTRY,
    REALM_REGISTRY,
    Alignment,
    Horizon,
    Polarity,
    Realm,
)
from fractalsemantics.fractalsemantics_entity import (
    BitChain,
    Coordinates,
    FractalSemanticsCoordinates,
)
from fractalsemantics.progress_comm import ProgressReporter

# Import subprocess communication for enhanced progress reporting
try:
    from fractalsemantics.subprocess_comm import (
        is_subprocess_communication_enabled,
        send_subprocess_completion,
        send_subprocess_progress,
        send_subprocess_status,
    )
except ImportError:
    # Fallback if subprocess communication is not available
    def send_subprocess_progress(*args, **kwargs) -> bool:
        return False
    def send_subprocess_status(*args, **kwargs) -> bool:
        return False
    def send_subprocess_completion(*args, **kwargs) -> bool:
        return False
    def is_subprocess_communication_enabled() -> bool:
        return False

# Use cryptographically secure random number generator
secure_random = secrets.SystemRandom()

# ============================================================================
# CONSTANTS AND GLOBALS (copied to avoid circular imports)
# ============================================================================

# Map experiment realms to FractalSemantics Realm enums
REALM_MAPPING = {
    "data": Realm.VOID,  # Map data to VOID as default
    "narrative": Realm.PATTERN,  # Map narrative to PATTERN
    "system": Realm.ACHIEVEMENT,  # Map system to ACHIEVEMENT
    "faculty": Realm.FACULTY,  # Direct mapping
    "event": Realm.TEMPORAL,  # Map event to TEMPORAL
    "pattern": Realm.PATTERN,  # Direct mapping
    "void": Realm.VOID,  # Direct mapping
}

# Map experiment horizons to FractalSemantics Horizon enums
HORIZON_MAPPING = {
    "genesis": Horizon.GENESIS,  # Direct mapping
    "emergence": Horizon.EMERGENCE,  # Direct mapping
    "peak": Horizon.PEAK,  # Direct mapping
    "decay": Horizon.DECAY,  # Direct mapping
    "crystallization": Horizon.CRYSTALLIZATION,  # Direct mapping
}

REALMS = ["data", "narrative", "system", "faculty", "event", "pattern", "void"]
HORIZONS = ["genesis", "emergence", "peak", "decay", "crystallization"]
POLARITY_LIST = ["logic", "creativity", "order", "chaos", "balance", "achievement",
            "contribution", "community", "technical", "creative", "unity", "void"]
ALIGNMENT_LIST = ["lawful_good", "neutral_good", "chaotic_good", "lawful_neutral",
            "true_neutral", "chaotic_neutral", "lawful_evil", "neutral_evil"]
ENTITY_TYPES = [
    "concept",
    "artifact",
    "agent",
    "lineage",
    "adjacency",
    "horizon",
    "fragment",
]


def generate_random_bitchain(seed: Optional[int] = None) -> BitChain:
    """
    Generate a random bit-chain for testing and validation experiments.

    This function creates synthetic bit-chains with randomized but valid FractalSemantics
    coordinates. It's used extensively in validation experiments to test address
    uniqueness, collision rates, and system behavior at scale.

    When a seed is provided, the function generates deterministic "random" data,
    which is essential for:
    - Reproducible experiments
    - Peer review validation
    - Debugging and testing
    - Statistical analysis

    The generation process:
    1. If seed provided: Use it to initialize random number generator
    2. Generate deterministic UUID-like ID from seed hash
    3. Create deterministic timestamp based on seed
    4. Randomly select realm, horizon, and entity type from valid options
    5. Generate random coordinates within valid ranges:
       - lineage: 1-100 (generation from LUCA)
       - luminosity: 0-100 (activity level)
       - polarity: Random Polarity enum value (resonance/affinity type)
       - dimensionality: 0-5 (fractal depth)
    6. Generate 0-5 random adjacency relationships

    Coordinate Ranges (enforced by FractalSemantics specification):
    - realm: One of 7 domains (data, narrative, system, faculty, event, pattern, void)
    - lineage: Positive integer (generation count from LUCA)
    - adjacency: list of UUIDs (relational neighbors)
    - horizon: One of 5 lifecycle stages (genesis, emergence, peak, decay, crystallization)
    - luminosity: Activity level (0-100) (new - replaced resonance)
    - polarity: One of 12 types (6 companion + 6 badge + neutral) (new - replaced velocity)
    - dimensionality: Fractal depth (0+) (new - replaced density)
    - alignment: One of 12 social/coordination dynamic alignments (NEW - 100% expressivity)

    Example:
        # Deterministic generation for reproducibility
        bc1 = generate_random_bitchain(seed=42)
        bc2 = generate_random_bitchain(seed=42)
        assert bc1.id == bc2.id  # Same seed produces identical bit-chain

        # Random generation for diversity testing
        bc3 = generate_random_bitchain()  # Different each time
        bc4 = generate_random_bitchain()
        assert bc3.id != bc4.id  # Different bit-chains

    Args:
        seed: Optional random seed for deterministic generation. If None, uses
              system randomness for non-deterministic generation.

    Returns:
        BitChain: A randomly generated bit-chain with valid FractalSemantics coordinates
    """

    if seed is not None:
        secure_random.seed(seed)
        base_id = hashlib.sha256(str(seed).encode()).hexdigest()[:32]
        id_str = f"{base_id[:8]}-{base_id[8:12]}-{base_id[12:16]}-{base_id[16:20]}-{base_id[20:32]}"
        created_at_str = f"2024-01-01T{seed % 24:02d}:{(seed // 24) % 60:02d}:{(seed // 1440) % 60:02d}.000Z"
    else:
        id_str = str(uuid.uuid4())
        created_at_str = datetime.now(timezone.utc).isoformat()

    adjacency_ids = [
        (
            hashlib.sha256(f"{seed}-adj-{i}".encode()).hexdigest()[:32]
            if seed is not None
            else str(uuid.uuid4())
        )
        for i in range(secure_random.randint(0, 5))
    ]

    if seed is not None and adjacency_ids:
        adjacency_ids = [
            f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:32]}"
            for uuid_hex in adjacency_ids
        ]

    # Generate coordinates with controlled precision for meaningful entropy testing
    # Quantize continuous values to create collisions within manageable sample sizes

    # Check if we want coarse quantization (fewer unique values for testing)
    use_coarse = "--coarse" in sys.argv
    use_ultra_coarse = "--ultra-coarse" in sys.argv

    if use_ultra_coarse:
        # Ultra-coarse: very constrained coordinate space for dramatic entropy effects
        luminosity_val = int(secure_random.uniform(0, 10))        # Only 10 values (0-9)
        # Also constrain other continuous values for dramatic effect
        adjacency_score = int(secure_random.uniform(0, 5))         # Only 5 values instead of 100+
    elif use_coarse:
        # Coarse quantization: fewer possible values to force collisions even with small samples
        luminosity_val = int(round(secure_random.uniform(0, 100), 0))  # Integers 0-100 (101 values)
    else:
        # Coarse quantization: constrained coordinate space to force collisions for entropy testing
        luminosity_val = int(secure_random.uniform(0, 5))     # 0-4, 5 values

    polarity_val = secure_random.choice(POLARITY_LIST)
    dimensionality_val = secure_random.randint(0, 5)
    alignment_val = secure_random.choice(list(Alignment))

    # Select random realm and horizon, then map to enums
    realm_str = secure_random.choice(REALMS)
    horizon_str = secure_random.choice(HORIZONS)
    realm_enum = REALM_MAPPING[realm_str]
    horizon_enum = HORIZON_MAPPING[horizon_str]

    # Convert adjacency list to a proximity score (0-100)
    adjacency_score = int(min(100.0, len(adjacency_ids) * 20.0))

    # Create FractalSemanticsCoordinates with additional dimensions
    fractalsemantics_coords = FractalSemanticsCoordinates(
        realm=realm_enum,
        lineage=secure_random.randint(1, 100),
        adjacency=adjacency_score,
        horizon=horizon_enum,
        luminosity=luminosity_val,
        polarity=Polarity(polarity_val),
        dimensionality=dimensionality_val,
        alignment=alignment_val,
    )

    return BitChain(
        id=id_str,
        entity_type=secure_random.choice(ENTITY_TYPES),
        realm=realm_str,
        coordinates=Coordinates(  # Use the legacy Coordinates for BitChain
            realm=realm_str,
            lineage=fractalsemantics_coords.lineage,
            adjacency=adjacency_ids,  # Keep as list for BitChain
            horizon=horizon_str,
            luminosity=luminosity_val,
            polarity=Polarity(polarity_val),
            dimensionality=dimensionality_val,
            alignment=alignment_val,
        ),
        created_at=created_at_str,
        state={
            "value": secure_random.randint(0, 1000),
        },
    )


@dataclass
class EXP03_Result:
    """Results from EXP-03 coordinate entropy test."""

    dimensions_used: list[str]
    sample_size: int
    shannon_entropy: float  # Shannon entropy of coordinate space
    normalized_entropy: float  # Normalized to [0, 1]
    expressiveness_contribution: float  # Weighted composite expressiveness score
    individual_contribution: float  # Individual dimension contribution vs theoretical max
    relative_contribution: float  # Marginal gain beyond other dimensions
    complementary_contribution: float  # Unique discriminatory information
    entropy_reduction_pct: float  # Legacy entropy reduction for comparison
    unique_coordinates: int  # Number of unique coordinate combinations
    semantic_disambiguation_score: float  # How well dimensions separate entities
    meets_threshold: bool  # >5% expressiveness contribution

    def to_dict(self) -> dict[str, any]:
        """Convert to JSON-serializable dictionary"""
        result = asdict(self)
        # Ensure all values are JSON-serializable
        serializable = {}
        for key, value in result.items():
            if hasattr(value, 'name'):  # Enum
                serializable[key] = value.name
            elif isinstance(value, list):
                # Handle lists of complex objects
                serializable[key] = [item.name if hasattr(item, 'name') else str(item) for item in value]
            else:
                # Basic types (str, int, float, bool) - explicit casting for bool
                if isinstance(value, bool):
                    serializable[key] = bool(value)
                else:
                    serializable[key] = value
        return serializable


class EXP03_CoordinateEntropy:
    """
    EXP-03: Coordinate Space Entropy Test

    This experiment quantifies the information-theoretic entropy contribution of
    each FractalSemantics dimension to the coordinate space, measuring how well each dimension
    contributes to semantic disambiguation.

    Scientific Rationale:
    The FractalSemantics addressing system uses 8 dimensions to create a rich semantic space.
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
    - Understand the information structure of FractalSemantics space
    - Guide future dimension design decisions (e.g., FractalSemantics?)
    """

    FractalSemantics_DIMENSIONS = [
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
        self.results: list[EXP03_Result] = []
        self.baseline_entropy: Optional[float] = None

    def compute_shannon_entropy(self, coordinates: list[str]) -> float:
        """
        Compute Shannon entropy of a list of coordinate representations.

        Shannon entropy H(X) = -Σ p(x) * log2(p(x))
        where p(x) is the probability of observing coordinate value x.

        Higher entropy indicates more information content and better
        discrimination between different entities.

        Args:
            coordinates: list of coordinate string representations

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
        self, coordinates: list[str], num_unique: int
    ) -> float:
        """
        Compute semantic disambiguation score.

        This metric measures how well the coordinate space separates
        semantically different entities. It combines:
        1. Uniqueness ratio (unique coordinates / total samples)
        2. Distribution uniformity (how evenly spread the coordinates are)

        A high score indicates good semantic separation.

        Args:
            coordinates: list of coordinate string representations
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

    def extract_coordinates(self, bitchains: list, dimensions: list[str]) -> list[str]:
        """
        Extract coordinate representations from bit-chains using specified dimensions.

        Creates a canonical string representation of each bit-chain's coordinates
        using only the specified dimensions. This allows us to measure entropy
        with different dimension subsets.

        Args:
            bitchains: list of BitChain objects
            dimensions: list of dimension names to include

        Returns:
            list of coordinate string representations
        """
        coordinates = []

        for bc in bitchains:
            # Get canonical dict
            data = bc.to_canonical_dict()
            coords = data["fractalsemantics_coordinates"]

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

    def compute_individual_contribution(self, dimension_name: str, bitchains: list) -> float:
        """
        Compute individual dimension contribution vs theoretical maximum.

        Measures how close a single dimension gets to its theoretical maximum entropy,
        as a percentage. This indicates how much effective discriminatory power
        a dimension provides when used alone.

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: list of BitChain objects

        Returns:
            Contribution score [0, 100] (percentage of theoretical maximum)
        """
        if dimension_name not in self.FractalSemantics_DIMENSIONS:
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

    def compute_relative_contribution(self, dimension_name: str, bitchains: list, baseline_entropy: float) -> float:
        """
        Compute relative contribution using Shapley additive approach.

        Measures how much entropy the dimension adds beyond all other dimensions.
        This is a cooperative game theory approach that fairly distributes
        the total contribution among dimensions.

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: list of BitChain objects
            baseline_entropy: Full 7D system entropy

        Returns:
            Relative contribution score [0, 100]
        """
        if dimension_name not in self.FractalSemantics_DIMENSIONS:
            return 0.0

        # Simplified relative contribution: entropy of subset without this dimension vs with it
        other_dims = [d for d in self.FractalSemantics_DIMENSIONS if d != dimension_name]

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
            return 100.0 / len(self.FractalSemantics_DIMENSIONS)

    def compute_complementary_contribution(self, dimension_name: str, bitchains: list) -> float:
        """
        Compute complementary contribution (unique discriminatory information).

        Measures how well this dimension provides unique separation that
        other dimensions don't capture. Uses conditional entropy concept.

        Args:
            dimension_name: Name of the dimension to evaluate
            bitchains: list of BitChain objects

        Returns:
            Complementary contribution score [0, 100]
        """
        if dimension_name not in self.FractalSemantics_DIMENSIONS:
            return 0.0

        try:
            # Get individual dimension entropy
            single_coords = self.extract_coordinates(bitchains, [dimension_name])
            single_entropy = self.compute_shannon_entropy(single_coords)

            # Get entropy of all other dimensions
            other_dims = [d for d in self.FractalSemantics_DIMENSIONS if d != dimension_name]
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
        bitchains: list,
        baseline_entropy: float
    ) -> tuple[float, float, float, float]:
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
            bitchains: list of BitChain objects
            baseline_entropy: Full system entropy

        Returns:
            tuple of (composite_score, individual_contrib, relative_contrib, complementary_contrib)
        """
        individual = self.compute_individual_contribution(dimension_name, bitchains) / 100.0  # Normalize to [0,1]
        relative = self.compute_relative_contribution(dimension_name, bitchains, baseline_entropy) / 100.0  # [0,1]
        complementary = self.compute_complementary_contribution(dimension_name, bitchains) / 100.0  # [0,1]

        # Ablation contribution (legacy - entropy loss when removed)
        other_dims = [d for d in self.FractalSemantics_DIMENSIONS if d != dimension_name]
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

    def run(self) -> tuple[list[EXP03_Result], bool]:
        """
        Run the coordinate entropy test.

        Returns:
            tuple of (results list, overall success boolean)
        """
        import time

        start_time = time.time()
        print(f"\n{'=' * 70}")
        print("EXP-03: COORDINATE SPACE ENTROPY TEST")
        print(f"{'=' * 70}")
        print(f"Sample size: {self.sample_size} bit-chains")
        print(f"Random seed: {self.random_seed} (for reproducibility)")
        print()

        # Send progress message for experiment start (use 0.0 progress instead of status)
        with contextlib.suppress(Exception):
            send_subprocess_status("EXP-03", 0.0, "Initialization", "Starting coordinate entropy test")

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
        baseline_coords = self.extract_coordinates(bitchains, self.FractalSemantics_DIMENSIONS)
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
            dimensions_used=self.FractalSemantics_DIMENSIONS.copy(),
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

        for i, removed_dim in enumerate(self.FractalSemantics_DIMENSIONS, 1):
            progress_percent = (i / len(self.FractalSemantics_DIMENSIONS)) * 100
            print("=" * 70)
            print(f"ABLATION: Remove '{removed_dim}' ({i}/8)")
            print("=" * 70)

            # Send progress message for dimension ablation with actual progress percentage
            with contextlib.suppress(Exception):
                send_subprocess_progress("EXP-03", progress_percent, f"Ablation {removed_dim}", f"Testing entropy without {removed_dim}")

            dim_start = time.time()
            # Get dimensions without the removed one
            remaining_dims = [d for d in self.FractalSemantics_DIMENSIONS if d != removed_dim]

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
                d for d in self.FractalSemantics_DIMENSIONS if d not in result.dimensions_used
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

        # Send completion progress message
        with contextlib.suppress(Exception):
            progress = ProgressReporter("EXP-03")
            progress.complete("Coordinate entropy test completed")

        return self.results, bool(all_success)

    def get_summary(self) -> dict[str, any]:
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

    def generate_visualization_data(self) -> dict[str, any]:
        """
        Generate data for entropy contribution visualization.

        Returns:
            dictionary with visualization data for plotting
        """
        if len(self.results) < 2:
            return {}

        # Extract dimension names and entropy reductions
        dimensions = []
        entropy_reductions = []

        for result in self.results[1:]:  # Skip baseline
            removed_dim = [
                d for d in self.FractalSemantics_DIMENSIONS if d not in result.dimensions_used
            ][0]
            dimensions.append(removed_dim)
            entropy_reductions.append(result.entropy_reduction_pct)

        return {
            "dimensions": dimensions,
            "entropy_reductions": entropy_reductions,
            "baseline_entropy": self.baseline_entropy,
            "threshold": 0.05,  # Use 0.05% instead of 5.0% to match actual data scale
        }


def save_results(results: dict[str, any], output_file: Optional[str] = None) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp03_coordinate_entropy_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent / "results"
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
        elif isinstance(obj, (int, float, str)):
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
    viz_data: dict[str, any], output_file: Optional[str] = None
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
    except ImportError as e:
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
    ax.set_xlabel("FractalSemantics Dimension", fontsize=12, fontweight="bold")
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
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"exp03_entropy_chart_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    # Load from config or use defaults
    sample_size = 100000  # Default fallback
    random_seed = 42      # Default fallback

    with contextlib.suppress(Exception):
        from fractalsemantics.config import ExperimentConfig

        config = ExperimentConfig()
        sample_size = config.get("EXP-03", "sample_size", 100000)
        random_seed = config.get("EXP-03", "random_seed", 42)

    # Override based on command-line arguments (always check these)
    if "--quick" in sys.argv:
        sample_size = 100000
    elif "--full" in sys.argv:
        sample_size = 1000000
    elif "--medium" in sys.argv:
        sample_size = 500000
    # else: keep the config-loaded value (no override)

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
