"""
EXP-01: Geometric Collision Resistance Test - Entities Module

This module defines the core data structures and entities used in the geometric
collision resistance experiment. These entities represent the results and
measurements of coordinate collision testing across different dimensional spaces.

The entities are designed to capture:
- Collision statistics across dimensional subspaces
- Coordinate space size calculations
- Geometric collision resistance validation metrics
- Experimental results and validation status

Author: FractalSemantics
Date: 2025-12-07
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class EXP01_Result:
    """
    Results from EXP-01 geometric collision resistance test.
    
    This dataclass captures the complete results of collision testing for a specific
    dimensional coordinate space, including collision rates, coordinate space sizes,
    and geometric validation status.
    
    Attributes:
        dimension: The number of dimensions being tested (2D, 3D, 4D, etc.)
        coordinate_space_size: Total possible coordinate combinations in this dimension
        sample_size: Number of coordinate samples generated for testing
        unique_coordinates: Number of unique coordinates found in the sample
        collisions: Number of coordinate collisions detected
        collision_rate: Percentage of collisions relative to sample size
        geometric_limit_hit: Whether sample size exceeded coordinate space bounds
    
    Scientific Significance:
        - dimension: Maps to coordinate space complexity and collision probability
        - coordinate_space_size: Theoretical maximum unique coordinates available
        - collision_rate: Primary metric for validating geometric collision resistance
        - geometric_limit_hit: Indicates when testing reaches mathematical limits
    """
    
    dimension: int
    coordinate_space_size: int
    sample_size: int
    unique_coordinates: int
    collisions: int
    collision_rate: float
    geometric_limit_hit: bool  # True if sample_size > coordinate_space

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the result suitable for JSON export
            
        Usage:
            result_dict = result.to_dict()
            json.dump(result_dict, file)
        """
        return asdict(self)

    def __str__(self) -> str:
        """
        String representation of the collision test result.
        
        Returns:
            Human-readable summary of the collision test results
            
        Example:
            "4D: 100,000 samples, 99,987 unique, 13 collisions (0.013%)"
        """
        return (
            f"{self.dimension}D: {self.sample_size:,} samples, "
            f"{self.unique_coordinates:,} unique, {self.collisions} collisions "
            f"({self.collision_rate * 100:.3f}%)"
        )

    def is_geometrically_resistant(self) -> bool:
        """
        Determine if this dimension exhibits geometric collision resistance.
        
        A dimension is considered geometrically resistant if:
        - It's 4D or higher AND
        - Collision rate is below 0.1% (indicating coordinate space is sufficiently large)
        
        Returns:
            True if geometric collision resistance is demonstrated
            
        Scientific Rationale:
            Higher dimensions should show exponentially lower collision rates due to
            the exponential growth of coordinate space size relative to sample size.
        """
        return self.dimension >= 4 and self.collision_rate < 0.001

    def get_collision_efficiency(self) -> float:
        """
        Calculate collision efficiency metric.
        
        Returns:
            Ratio of unique coordinates to total samples (higher is better)
            
        Usage:
            efficiency = result.get_collision_efficiency()  # 0.0 to 1.0
        """
        return self.unique_coordinates / self.sample_size if self.sample_size > 0 else 0.0