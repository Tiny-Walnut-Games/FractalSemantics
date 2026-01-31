"""
EXP-16: Hierarchical Distance to Euclidean Distance Mapping - Data Entities

This module contains all the data structures and entities used in the hierarchical
distance mapping experiment. These entities represent the core components for testing
whether hierarchical distance (discrete tree hops) maps to Euclidean distance
(continuous spatial distance) through fractal embedding strategies.

CORE HYPOTHESIS:
When fractal hierarchies are embedded in Euclidean space, hierarchical distance
d_h (tree hops) relates to Euclidean distance r through a power-law: r ∝ d_h^exponent.

Classes:
- EmbeddedFractalHierarchy: Fractal hierarchy embedded in Euclidean space
- EmbeddingStrategy: Base class for embedding strategies
- ExponentialEmbedding: Exponential distance scaling embedding
- SphericalEmbedding: Spherical shell-based embedding
- RecursiveEmbedding: Recursive space partitioning embedding
- DistancePair: Pair of nodes with both distance measurements
- DistanceMappingAnalysis: Analysis of distance mapping relationships
- ForceScalingValidation: Validation of force scaling consistency
"""

import json
import time
import secrets
import sys
import random
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import math
import statistics

# Import from EXP-13 for fractal hierarchy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from exp13_fractal_gravity import (
    FractalHierarchy,
    FractalNode,
)

secure_random = secrets.SystemRandom()

# ============================================================================
# EXP-16: EMBEDDED FRACTAL HIERARCHY
# ============================================================================

@dataclass
class EmbeddedFractalHierarchy:
    """
    A fractal hierarchy embedded in Euclidean space using a specific strategy.
    """
    hierarchy: FractalHierarchy
    embedding_type: str
    positions: Dict[FractalNode, np.ndarray]  # node -> [x,y,z] position
    scale_factor: float = 1.0

    def get_euclidean_distance(self, node_a: FractalNode, node_b: FractalNode) -> float:
        """Get Euclidean distance between two embedded nodes."""
        pos_a = self.positions[node_a]
        pos_b = self.positions[node_b]
        return np.linalg.norm(pos_a - pos_b)

    def get_hierarchical_distance(self, node_a: FractalNode, node_b: FractalNode) -> int:
        """Get hierarchical distance between two nodes."""
        return self.hierarchy.get_hierarchical_distance(node_a, node_b)


@dataclass
class EmbeddingStrategy:
    """
    A strategy for embedding fractal hierarchies in Euclidean space.
    """
    name: str
    description: str

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """
        Embed the given hierarchy using this strategy.

        Args:
            hierarchy: The fractal hierarchy to embed
            scale_factor: Scaling factor for the embedding

        Returns:
            Embedded hierarchy with positions
        """
        raise NotImplementedError("Subclasses must implement embed_hierarchy")


class ExponentialEmbedding(EmbeddingStrategy):
    """
    Exponential embedding: Nodes placed at exponentially increasing distances from root.
    """

    def __init__(self):
        super().__init__(
            name="Exponential",
            description="Nodes placed at exponential distance from parent (d ∝ α^depth)"
        )

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """Embed using exponential distance scaling."""
        positions = {}

        # Root node at origin
        root_nodes = hierarchy.nodes_by_depth[0]
        if root_nodes:
            positions[root_nodes[0]] = np.array([0.0, 0.0, 0.0])

        # Place nodes level by level
        for depth in range(1, hierarchy.max_depth):
            parent_depth = depth - 1
            if parent_depth not in hierarchy.nodes_by_depth:
                continue

            parent_nodes = hierarchy.nodes_by_depth[parent_depth]
            child_nodes = hierarchy.nodes_by_depth.get(depth, [])

            # Group children by parent
            children_by_parent = {}
            for child in child_nodes:
                parent_idx = child.tree_address[0] if child.tree_address else 0
                if parent_idx < len(parent_nodes):
                    parent = parent_nodes[parent_idx]
                    if parent not in children_by_parent:
                        children_by_parent[parent] = []
                    children_by_parent[parent].append(child)

            # Place children around each parent
            for parent, children in children_by_parent.items():
                parent_pos = positions[parent]
                radius = scale_factor * (2.0 ** depth)  # Exponential scaling

                for i, child in enumerate(children):
                    # Distribute children in a circle around parent
                    angle = 2 * np.pi * i / len(children)
                    offset = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        scale_factor * depth * 0.1  # Small vertical offset
                    ])
                    positions[child] = parent_pos + offset

        return EmbeddedFractalHierarchy(
            hierarchy=hierarchy,
            embedding_type=self.name,
            positions=positions,
            scale_factor=scale_factor
        )


class SphericalEmbedding(EmbeddingStrategy):
    """
    Spherical embedding: Nodes placed on concentric spheres.
    """

    def __init__(self):
        super().__init__(
            name="Spherical",
            description="Nodes placed on concentric spheres (shell-based)"
        )

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """Embed using spherical shells."""
        positions = {}

        # Root at origin
        root_nodes = hierarchy.nodes_by_depth[0]
        if root_nodes:
            positions[root_nodes[0]] = np.array([0.0, 0.0, 0.0])

        # Place nodes on spherical shells by depth
        for depth in range(1, hierarchy.max_depth):
            nodes_at_depth = hierarchy.nodes_by_depth.get(depth, [])
            if not nodes_at_depth:
                continue

            # Radius increases with depth
            radius = scale_factor * (depth + 1)

            # Distribute nodes evenly on sphere
            for i, node in enumerate(nodes_at_depth):
                # Golden spiral distribution for even spacing
                golden_ratio = (1 + np.sqrt(5)) / 2
                theta = 2 * np.pi * i / golden_ratio  # Longitude
                phi = np.arccos(1 - 2 * (i + 0.5) / len(nodes_at_depth))  # Latitude

                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)

                positions[node] = np.array([x, y, z])

        return EmbeddedFractalHierarchy(
            hierarchy=hierarchy,
            embedding_type=self.name,
            positions=positions,
            scale_factor=scale_factor
        )


class RecursiveEmbedding(EmbeddingStrategy):
    """
    Recursive embedding: Each branching splits space along coordinate axes.
    """

    def __init__(self):
        super().__init__(
            name="Recursive",
            description="Recursive space division along coordinate axes"
        )

    def embed_hierarchy(
        self,
        hierarchy: FractalHierarchy,
        scale_factor: float = 1.0
    ) -> EmbeddedFractalHierarchy:
        """Embed using recursive space partitioning."""
        positions = {}

        def place_node_recursive(node: FractalNode, center: np.ndarray, size: float):
            """Recursively place a node and its children."""
            positions[node] = center.copy()

            # Get children
            child_depth = node.hierarchical_depth + 1
            if child_depth >= hierarchy.max_depth:
                return

            children = []
            for potential_child in hierarchy.nodes_by_depth.get(child_depth, []):
                if (potential_child.tree_address and
                    len(potential_child.tree_address) > node.hierarchical_depth and
                    potential_child.tree_address[node.hierarchical_depth] == node.tree_address[-1] if node.tree_address else 0):
                    children.append(potential_child)

            if not children:
                return

            # Split space among children
            child_size = size / 2.0
            axes = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]  # 6 directions

            for i, child in enumerate(children):
                if i < len(axes):
                    direction = np.array(axes[i])
                    child_center = center + direction * child_size * scale_factor
                    place_node_recursive(child, child_center, child_size)

        # Start with root
        root_nodes = hierarchy.nodes_by_depth[0]
        if root_nodes:
            place_node_recursive(root_nodes[0], np.array([0.0, 0.0, 0.0]), scale_factor)

        return EmbeddedFractalHierarchy(
            hierarchy=hierarchy,
            embedding_type=self.name,
            positions=positions,
            scale_factor=scale_factor
        )


# ============================================================================
# EXP-16: DISTANCE MEASUREMENT AND ANALYSIS
# ============================================================================

@dataclass
class DistancePair:
    """A pair of nodes with both distance measurements."""
    node_a: FractalNode
    node_b: FractalNode
    hierarchical_distance: int
    euclidean_distance: float

    @property
    def distance_ratio(self) -> float:
        """Ratio of Euclidean to hierarchical distance."""
        return self.euclidean_distance / max(1, self.hierarchical_distance)


@dataclass
class DistanceMappingAnalysis:
    """
    Analysis of distance mapping for an embedded hierarchy.
    """
    embedding: EmbeddedFractalHierarchy
    distance_pairs: List[DistancePair]

    # Power-law fitting results
    power_law_exponent: float = field(init=False)
    power_law_coefficient: float = field(init=False)
    correlation_coefficient: float = field(init=False)
    mapping_quality: float = field(init=False)

    def __post_init__(self):
        """Fit power-law relationship between distances."""
        if not self.distance_pairs:
            self.power_law_exponent = 0.0
            self.power_law_coefficient = 0.0
            self.correlation_coefficient = 0.0
            self.mapping_quality = 0.0
            return

        # Extract distance data
        h_distances = [p.hierarchical_distance for p in self.distance_pairs]
        e_distances = [p.euclidean_distance for p in self.distance_pairs]

        # Fit power law: e_distance = coefficient * h_distance^exponent
        try:
            # Use log-linear regression
            log_h = [math.log(max(1, d)) for d in h_distances]
            log_e = [math.log(max(1e-6, d)) for d in e_distances]

            # Linear regression on log scales
            n = len(log_h)
            sum_x = sum(log_h)
            sum_y = sum(log_e)
            sum_xy = sum(x*y for x,y in zip(log_h, log_e))
            sum_x2 = sum(x*x for x in log_h)

            # Slope (exponent) and intercept (log coefficient)
            denominator = n * sum_x2 - sum_x**2
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / n

                self.power_law_exponent = slope
                self.power_law_coefficient = math.exp(intercept)

                # Calculate correlation
                y_pred = [intercept + slope * x for x in log_h]
                correlation = np.corrcoef(log_e, y_pred)[0, 1]
                self.correlation_coefficient = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                self.power_law_exponent = 1.0
                self.power_law_coefficient = 1.0
                self.correlation_coefficient = 0.0

        except Exception:
            self.power_law_exponent = 1.0
            self.power_law_coefficient = 1.0
            self.correlation_coefficient = 0.0

        # Overall mapping quality (0-1 scale)
        self.mapping_quality = min(1.0, self.correlation_coefficient)


@dataclass
class ForceScalingValidation:
    """
    Validation of force scaling consistency between hierarchical and Euclidean approaches.
    """
    embedding_analysis: DistanceMappingAnalysis
    hierarchical_forces: List[float]
    euclidean_forces: List[float]

    # Force scaling analysis
    force_correlation: float = field(init=False)
    scaling_consistency: float = field(init=False)

    def __post_init__(self):
        """Analyze force scaling consistency."""
        if not self.hierarchical_forces or not self.euclidean_forces:
            self.force_correlation = 0.0
            self.scaling_consistency = 0.0
            return

        try:
            correlation = np.corrcoef(self.hierarchical_forces, self.euclidean_forces)[0, 1]
            self.force_correlation = abs(correlation) if not np.isnan(correlation) else 0.0
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not compute force correlation: {e}")
            self.force_correlation = 0.0

        # Scaling consistency: how well the relative force magnitudes match
        self.scaling_consistency = self.force_correlation