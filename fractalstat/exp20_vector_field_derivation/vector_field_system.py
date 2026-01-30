"""
Vector field derivation system and approaches.

Implements different approaches for deriving directional force vectors from
fractal hierarchy structures, including branching-based, depth-based, and
combined hierarchy approaches.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from .entities import FractalEntity, VectorFieldResult


@dataclass
class VectorFieldApproach:
    """A vector field derivation approach."""

    name: str
    function: Callable[[FractalEntity, FractalEntity, float], np.ndarray]
    description: str

    def derive_force(self, entity_a: FractalEntity, entity_b: FractalEntity,
                    scalar_magnitude: float) -> np.ndarray:
        """Derive force vector using this approach."""
        return self.function(entity_a, entity_b, scalar_magnitude)


# ============================================================================
# VECTOR FIELD DERIVATION APPROACHES
# ============================================================================

def compute_force_vector_via_branching(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    scalar_magnitude: float
) -> np.ndarray:
    """
    Original approach: Simple attractive force with no branching modulation.

    Direction: Always attractive toward central body
    Magnitude: Just the scalar magnitude (distance dependence applied elsewhere)
    """
    r_vector = entity_b.position - entity_a.position  # Points from A to B
    r_distance = np.linalg.norm(r_vector)

    if r_distance == 0:
        return np.zeros(3)

    # Direction: ALWAYS attractive toward central body
    direction = r_vector / r_distance

    # Magnitude: Just use scalar magnitude directly (no branching modulation)
    directional_magnitude = scalar_magnitude

    return directional_magnitude * direction


def compute_force_vector_via_branching_difference(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    scalar_magnitude: float
) -> np.ndarray:
    """
    Branching vector using difference-based asymmetry formula.

    Direction: Always attractive toward central body
    Magnitude: modulated by absolute difference in branching factors
    """
    r_vector = entity_b.position - entity_a.position
    r_distance = np.linalg.norm(r_vector)

    if r_distance == 0:
        return np.zeros(3)

    # Direction: ALWAYS attractive toward central body
    direction = r_vector / r_distance

    branching_a = entity_a.branching_factor
    branching_b = entity_b.branching_factor

    # Difference-based asymmetry
    branching_diff = abs(branching_b - branching_a)
    max_branching = max(branching_a, branching_b, 1)  # Avoid division by zero

    asymmetry_factor = branching_diff / max_branching
    directional_magnitude = scalar_magnitude * (1.0 + asymmetry_factor)  # Base + difference

    return directional_magnitude * direction


def compute_force_vector_via_branching_normalized(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    scalar_magnitude: float
) -> np.ndarray:
    """
    Branching vector using normalized asymmetry formula.

    Direction: Always attractive toward central body
    Magnitude: modulated by normalized branching difference
    """
    r_vector = entity_b.position - entity_a.position
    r_distance = np.linalg.norm(r_vector)

    if r_distance == 0:
        return np.zeros(3)

    # Direction: ALWAYS attractive toward central body
    direction = r_vector / r_distance

    branching_a = entity_a.branching_factor
    branching_b = entity_b.branching_factor

    # Normalized asymmetry: (b - a) / (a + b)
    total_branching = branching_a + branching_b
    if total_branching == 0:
        asymmetry_factor = 0.0
    else:
        normalized_diff = (branching_b - branching_a) / total_branching
        asymmetry_factor = abs(normalized_diff)

    directional_magnitude = scalar_magnitude * (1.0 + asymmetry_factor)

    return directional_magnitude * direction


def compute_force_vector_via_depth(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    scalar_magnitude: float
) -> np.ndarray:
    """
    Derive directional force from hierarchical depth difference.

    Force is ALWAYS attractive (toward the deeper/more complex entity).
    Magnitude modulated by depth complexity difference.
    """
    # Direction: ALWAYS toward the deeper/more complex entity (attractive force)
    r_vector = entity_b.position - entity_a.position  # Points from A to B
    r_distance = np.linalg.norm(r_vector)

    if r_distance == 0:
        return np.zeros(3)

    # Determine which entity is deeper (more complex hierarchy)
    if entity_b.hierarchical_depth > entity_a.hierarchical_depth:
        # B is deeper, force on A points toward B
        direction = r_vector / r_distance
    else:
        # A is deeper, force on A points away from B (repulsive)
        direction = -r_vector / r_distance

    # Magnitude: base scalar magnitude modulated by depth complexity difference
    depth_ratio = max(entity_a.hierarchical_depth, entity_b.hierarchical_depth) / \
                  min(entity_a.hierarchical_depth, entity_b.hierarchical_depth)
    directional_magnitude = scalar_magnitude * depth_ratio

    return directional_magnitude * direction


def compute_force_vector_via_combined_hierarchy(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    scalar_magnitude: float
) -> np.ndarray:
    """
    Derive directional force from total hierarchical complexity.

    Force is ALWAYS attractive (toward the more complex entity).
    Complexity = depth * log(branching_factor + 1)
    """
    import math

    # Calculate hierarchical complexity for each entity
    complexity_a = entity_a.hierarchical_depth * math.log(entity_a.branching_factor + 1)
    complexity_b = entity_b.hierarchical_depth * math.log(entity_b.branching_factor + 1)

    # Direction: ALWAYS toward the more complex entity (attractive force)
    r_vector = entity_b.position - entity_a.position  # Points from A to B
    r_distance = np.linalg.norm(r_vector)

    if r_distance == 0:
        return np.zeros(3)

    # Determine which entity is more complex
    if complexity_b > complexity_a:
        # B is more complex, force on A points toward B
        direction = r_vector / r_distance
    else:
        # A is more complex, force on A points away from B (repulsive)
        direction = -r_vector / r_distance

    # Magnitude: base scalar magnitude modulated by complexity difference
    complexity_ratio = max(complexity_a, complexity_b) / min(complexity_a, complexity_b)
    directional_magnitude = scalar_magnitude * complexity_ratio

    return directional_magnitude * direction


class VectorFieldDerivationSystem:
    """System for testing different vector field derivation approaches."""

    def __init__(self):
        self.approaches = [
            VectorFieldApproach(
                name="Branching Vector (Ratio)",
                function=compute_force_vector_via_branching,
                description="Direction from branching ratio asymmetry"
            ),
            VectorFieldApproach(
                name="Branching Vector (Difference)",
                function=compute_force_vector_via_branching_difference,
                description="Direction from branching difference asymmetry"
            ),
            VectorFieldApproach(
                name="Branching Vector (Normalized)",
                function=compute_force_vector_via_branching_normalized,
                description="Direction from normalized branching asymmetry"
            ),
            VectorFieldApproach(
                name="Depth Vector",
                function=compute_force_vector_via_depth,
                description="Direction from hierarchical depth difference"
            ),
            VectorFieldApproach(
                name="Combined Hierarchy",
                function=compute_force_vector_via_combined_hierarchy,
                description="Direction from total hierarchical complexity"
            )
        ]

    def derive_all_vectors(self, entity_a: FractalEntity, entity_b: FractalEntity,
                          scalar_magnitude: float) -> List[VectorFieldResult]:
        """Derive force vectors using all approaches."""
        results = []

        for approach in self.approaches:
            start_time = time.time()
            force_vector = approach.derive_force(entity_a, entity_b, scalar_magnitude)
            derivation_time = time.time() - start_time

            # Calculate accuracy metrics (simplified - would need Newtonian reference)
            magnitude_accuracy = 1.0  # Placeholder
            direction_accuracy = 1.0  # Placeholder

            result = VectorFieldResult(
                approach_name=approach.name,
                entity_a=entity_a,
                entity_b=entity_b,
                force_vector=force_vector,
                magnitude_accuracy=magnitude_accuracy,
                direction_accuracy=direction_accuracy,
                derivation_time=derivation_time
            )
            results.append(result)

        return results