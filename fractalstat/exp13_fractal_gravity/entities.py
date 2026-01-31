"""
EXP-13: Fractal Gravity - Data Entities

This module contains all the data structures and entities used in the fractal gravity
experiment. These entities represent the core components of the hierarchical fractal
system including nodes, hierarchies, cohesion measurements, and experimental results.

Classes:
- FractalNode: A node in a pure fractal hierarchy (no spatial coordinates)
- FractalHierarchy: A pure fractal tree structure for a single element type
- HierarchicalCohesionMeasurement: Records cohesion between nodes at specific distances
- ElementGravityResults: Results for a single element's gravitational behavior
- EXP13v2_GravityTestResults: Complete results from fractal gravity test
"""

import json
import time
import secrets
import sys
import random
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import statistics

secure_random = secrets.SystemRandom()

# ============================================================================
# EXP-13 v2: PURE HIERARCHICAL DATA STRUCTURES
# ============================================================================

@dataclass
class FractalNode:
    """A node in a pure fractal hierarchy - NO spatial coordinates."""

    element: str  # Element type (gold, nickel, etc.)
    hierarchical_depth: int  # Depth in fractal tree (0 = root, 1 = child, etc.)
    tree_address: List[int]  # Address in tree (e.g., [0, 2, 1] = grandchild of second child of root)

    def __repr__(self):
        return f"Node({self.element}, depth={self.hierarchical_depth}, addr={self.tree_address})"

    def __hash__(self):
        """Make FractalNode hashable for use as dictionary keys."""
        return hash((self.element, self.hierarchical_depth, tuple(self.tree_address)))

    def __eq__(self, other):
        """Check equality for hash consistency."""
        if not isinstance(other, FractalNode):
            return False
        return (self.element == other.element and
                self.hierarchical_depth == other.hierarchical_depth and
                self.tree_address == other.tree_address)


@dataclass
class FractalHierarchy:
    """A pure fractal tree structure for a single element type."""

    element: str
    max_depth: int
    branching_factor: int
    nodes_by_depth: Dict[int, List[FractalNode]]

    @classmethod
    def build(cls, element_type: str, max_depth: int, branching_factor: int = 3):
        """Build a complete fractal hierarchy tree."""
        instance = cls(
            element=element_type,
            max_depth=max_depth,
            branching_factor=branching_factor,
            nodes_by_depth={0: [FractalNode(element_type, 0, [])]}
        )

        # Build tree level by level
        for depth in range(1, max_depth):
            instance.nodes_by_depth[depth] = []
            parent_count = len(instance.nodes_by_depth[depth - 1])

            for parent_idx in range(parent_count):
                # Each parent has branching_factor children
                for child_idx in range(branching_factor):
                    child_addr = [parent_idx, child_idx]
                    child = FractalNode(element_type, depth, child_addr)
                    instance.nodes_by_depth[depth].append(child)

        return instance

    def get_all_nodes(self) -> List[FractalNode]:
        """Get all nodes in the hierarchy."""
        all_nodes = []
        for nodes in self.nodes_by_depth.values():
            all_nodes.extend(nodes)
        return all_nodes

    def get_hierarchical_distance(self, node_a: FractalNode, node_b: FractalNode) -> int:
        """
        Calculate hierarchical distance as hops through tree to lowest common ancestor.

        Example: if A is at depth 3 and B is at depth 3, but they share
        a parent at depth 2, the hierarchical distance is 2 (A→parent→B)
        """
        addr_a = node_a.tree_address
        addr_b = node_b.tree_address

        # Find lowest common ancestor depth
        common_depth = 0
        for i in range(min(len(addr_a), len(addr_b))):
            if addr_a[i] == addr_b[i]:
                common_depth = i + 1
            else:
                break

        # Distance = hops up to ancestor + hops down
        distance = (len(addr_a) - common_depth) + (len(addr_b) - common_depth)
        return max(1, distance)


@dataclass
class HierarchicalCohesionMeasurement:
    """Records cohesion measurement between two nodes at a specific hierarchical distance."""

    hierarchical_distance: int
    node_a: FractalNode
    node_b: FractalNode
    natural_cohesion: float  # Without falloff - should be CONSTANT
    falloff_cohesion: float  # With falloff - should follow pattern


@dataclass
class ElementGravityResults:
    """Results for a single element's gravitational behavior."""

    element: str
    total_measurements: int
    cohesion_by_hierarchical_distance: Dict[int, Dict[str, Any]]
    element_fractal_density: float  # Derived property

    # Key metrics (calculated in __post_init__)
    natural_cohesion_flatness: float = field(init=False)  # 1.0 = perfectly constant across distances
    falloff_pattern_consistency: float = field(init=False)  # 1.0 = follows inverse-square perfectly

    def __post_init__(self):
        """Calculate derived metrics."""
        self._calculate_flatness()
        self._calculate_pattern_consistency()

    def _calculate_flatness(self):
        """Calculate how constant natural cohesion is across hierarchical distances."""
        if not self.cohesion_by_hierarchical_distance:
            self.natural_cohesion_flatness = 0.0
            return

        means = []
        for dist_data in self.cohesion_by_hierarchical_distance.values():
            if 'natural' in dist_data and dist_data['natural']['measurements']:
                means.append(dist_data['natural']['mean'])

        if len(means) <= 1:
            self.natural_cohesion_flatness = 1.0  # Single distance = perfectly flat
        else:
            # Flatness = 1 - (coefficient of variation of means)
            mean_of_means = statistics.mean(means)
            std_of_means = statistics.stdev(means) if len(means) > 1 else 0

            if mean_of_means > 0:
                self.natural_cohesion_flatness = max(0.0, 1.0 - (std_of_means / mean_of_means))
            else:
                self.natural_cohesion_flatness = 0.0

    def _calculate_pattern_consistency(self):
        """Calculate how well falloff follows inverse-square pattern."""
        if not self.cohesion_by_hierarchical_distance:
            self.falloff_pattern_consistency = 0.0
            return

        # Test against theoretical 1/distance^2 pattern
        distances = []
        measured_means = []

        for h_dist, dist_data in self.cohesion_by_hierarchical_distance.items():
            if 'falloff' in dist_data and dist_data['falloff']['measurements']:
                distances.append(h_dist)
                measured_means.append(dist_data['falloff']['mean'])

        if len(distances) < 2:
            self.falloff_pattern_consistency = 0.5  # Insufficient data
            return

        # Calculate theoretical values: base_cohesion / distance^2
        # Use first distance as reference
        ref_distance = min(distances)
        ref_mean = None
        for d, m in zip(distances, measured_means):
            if d == ref_distance:
                ref_mean = m
                break

        if ref_mean is None or ref_mean <= 0:
            self.falloff_pattern_consistency = 0.0
            return

        theoretical_values = []
        for d in distances:
            theoretical = ref_mean / (d ** 2)
            theoretical_values.append(theoretical)

        # Calculate correlation between measured and theoretical
        try:
            correlation = np.corrcoef(measured_means, theoretical_values)[0, 1]
            self.falloff_pattern_consistency = max(0.0, min(1.0, abs(correlation)))
        except (ValueError, TypeError, IndexError) as e:
            # Handle cases where correlation cannot be computed
            print(f"Warning: Could not compute falloff pattern consistency: {e}")
            self.falloff_pattern_consistency = 0.0


@dataclass
class EXP13v2_GravityTestResults:
    """Complete results from EXP-13 v2 fractal gravity test."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    elements_tested: List[str]
    element_results: Dict[str, ElementGravityResults]
    measurements: List[HierarchicalCohesionMeasurement]

    # Cross-element analysis
    fractal_no_falloff_confirmed: bool  # Natural cohesion is constant across hierarchy
    universal_falloff_mechanism: bool   # Same falloff pattern for all elements
    mass_fractal_density_correlation: float  # How well mass correlates with fractal properties