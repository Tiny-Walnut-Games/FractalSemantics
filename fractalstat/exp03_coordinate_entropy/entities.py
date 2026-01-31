"""
EXP-03: Coordinate Space Entropy Test - Entities Module

This module defines the core data structures and entities used in the coordinate
entropy experiment. These entities capture information-theoretic measurements,
entropy calculations, and semantic disambiguation metrics across different
dimensional configurations.

The entities are designed to capture:
- Shannon entropy measurements at coordinate level
- Information-theoretic contribution analysis
- Semantic disambiguation power metrics
- Ablation study results for dimension necessity
- Comprehensive expressiveness scoring

Author: FractalSemantics
Date: 2025-12-07
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


@dataclass
class EXP03_Result:
    """
    Results from EXP-03 coordinate entropy test.
    
    This dataclass captures comprehensive information-theoretic measurements
    for coordinate space entropy analysis across different dimensional configurations.
    
    Attributes:
        dimensions_used: List of dimensions included in this measurement
        sample_size: Number of bit-chains used for entropy calculation
        shannon_entropy: Shannon entropy of coordinate space in bits
        normalized_entropy: Entropy normalized to [0, 1] range
        expressiveness_contribution: Weighted composite expressiveness score (0-100%)
        individual_contribution: Individual dimension contribution vs theoretical max (0-100%)
        relative_contribution: Marginal gain beyond other dimensions (0-100%)
        complementary_contribution: Unique discriminatory information provided (0-100%)
        entropy_reduction_pct: Percentage reduction in entropy when dimension removed
        unique_coordinates: Number of unique coordinate combinations
        semantic_disambiguation_score: How well dimensions separate entities (0-1)
        meets_threshold: Whether dimension meets >5% expressiveness contribution threshold
    
    Information-Theoretic Metrics:
        - Shannon Entropy: Measures information content of coordinate space
        - Normalized Entropy: Scales entropy to comparable [0, 1] range
        - Expressiveness Contribution: Composite score combining multiple approaches
        - Semantic Disambiguation: Measures separation of semantically different entities
    """
    
    dimensions_used: List[str]
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.
        
        Handles complex types like enums and ensures all values are JSON-compatible.
        
        Returns:
            Dictionary representation suitable for JSON export
            
        Usage:
            result_dict = result.to_dict()
            json.dump(result_dict, file)
        """
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

    def __str__(self) -> str:
        """
        String representation of the coordinate entropy test result.
        
        Returns:
            Human-readable summary of the entropy analysis results
            
        Example:
            "7D (remove alignment): entropy=12.345 bits, reduction=15.2%, critical=True"
        """
        dims_str = f"{len(self.dimensions_used)}D"
        removed_dims = [d for d in ["realm", "lineage", "adjacency", "horizon", "luminosity", "polarity", "dimensionality", "alignment"] 
                       if d not in self.dimensions_used]
        if removed_dims:
            dims_str += f" (remove {', '.join(removed_dims)})"
        
        return (
            f"{dims_str}: entropy={self.shannon_entropy:.3f} bits, "
            f"reduction={self.entropy_reduction_pct:.1f}%, "
            f"critical={self.meets_threshold}"
        )

    def get_entropy_analysis(self) -> Dict[str, float]:
        """
        Get comprehensive entropy analysis metrics.
        
        Returns:
            Dictionary containing all entropy-related measurements
            
        Analysis Includes:
            - Raw Shannon entropy in bits
            - Normalized entropy for comparison across scales
            - Entropy reduction percentage vs baseline
            - Unique coordinate count and ratio
        """
        return {
            'shannon_entropy_bits': self.shannon_entropy,
            'normalized_entropy': self.normalized_entropy,
            'entropy_reduction_percentage': self.entropy_reduction_pct,
            'unique_coordinates': self.unique_coordinates,
            'coordinate_uniqueness_ratio': self.unique_coordinates / self.sample_size if self.sample_size > 0 else 0.0,
            'semantic_disambiguation_score': self.semantic_disambiguation_score
        }

    def get_expressiveness_breakdown(self) -> Dict[str, float]:
        """
        Get detailed breakdown of expressiveness contributions.
        
        Returns:
            Dictionary with individual contribution metrics
            
        Contribution Types:
            - Individual: How close to theoretical maximum alone
            - Relative: Marginal gain beyond other dimensions
            - Complementary: Unique information not captured by others
            - Composite: Weighted combination of all approaches
        """
        return {
            'individual_contribution_percentage': self.individual_contribution,
            'relative_contribution_percentage': self.relative_contribution,
            'complementary_contribution_percentage': self.complementary_contribution,
            'composite_expressiveness_percentage': self.expressiveness_contribution,
            'meets_critical_threshold': self.meets_threshold
        }

    def is_critical_dimension(self) -> bool:
        """
        Determine if this dimension configuration reveals critical dimensions.
        
        A dimension is considered critical if its removal causes significant
        entropy reduction (>5% expressiveness contribution).
        
        Returns:
            True if dimension removal shows critical impact on entropy
        """
        return self.meets_threshold

    def get_dimension_impact(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of dimension impact.
        
        Returns:
            Dictionary with complete impact analysis including entropy,
            expressiveness, and disambiguation metrics
            
        Impact Analysis:
            - Information loss when dimension removed
            - Discriminatory power reduction
            - Semantic separation degradation
            - Overall system expressiveness impact
        """
        removed_dims = [d for d in ["realm", "lineage", "adjacency", "horizon", "luminosity", "polarity", "dimensionality", "alignment"] 
                       if d not in self.dimensions_used]
        
        return {
            'removed_dimensions': removed_dims,
            'remaining_dimensions': self.dimensions_used,
            'entropy_impact': {
                'shannon_entropy': self.shannon_entropy,
                'normalized_entropy': self.normalized_entropy,
                'entropy_reduction': self.entropy_reduction_pct
            },
            'expressiveness_impact': self.get_expressiveness_breakdown(),
            'disambiguation_impact': {
                'semantic_score': self.semantic_disambiguation_score,
                'unique_coordinates': self.unique_coordinates,
                'coordinate_ratio': self.unique_coordinates / self.sample_size if self.sample_size > 0 else 0.0
            },
            'criticality_assessment': {
                'is_critical': self.is_critical_dimension(),
                'impact_level': 'HIGH' if self.expressiveness_contribution > 20.0 else 'MEDIUM' if self.expressiveness_contribution > 5.0 else 'LOW'
            }
        }