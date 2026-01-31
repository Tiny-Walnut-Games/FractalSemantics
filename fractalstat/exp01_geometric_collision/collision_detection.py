"""
EXP-01: Geometric Collision Resistance Test - Collision Detection

Implements the core collision detection algorithms and coordinate space testing
for the geometric collision resistance experiment. This module handles the
empirical testing of collision rates across different dimensionalities.

Scientific Rationale:
The collision detection system validates that FractalStat coordinate space
exhibits mathematical collision resistance properties independent of cryptographic hashing.

Key Features:
- Uniform coordinate generation across dimensionalities
- Collision tracking and rate calculation
- Geometric limit detection
- Performance optimization for large-scale testing
"""

import sys
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime
from .entities import CoordinateSpaceAnalyzer, EXP01_Result


class CollisionDetector:
    """
    Core collision detection system for geometric collision resistance testing.
    
    This class implements the empirical testing of collision rates across
    different dimensional coordinate spaces, providing the mathematical validation
    that FractalStat coordinates achieve collision resistance through geometry.
    """

    def __init__(self, sample_size: int = 100000):
        """
        Initialize collision detector with specified sample size.
        
        Args:
            sample_size: Number of coordinate samples to generate per dimension
        """
        self.sample_size = sample_size
        self.analyzer = CoordinateSpaceAnalyzer()

    def test_dimension(self, dimension: int) -> EXP01_Result:
        """
        Test collision resistance for a specific dimension.
        
        Args:
            dimension: Dimensionality of coordinate space to test
            
        Returns:
            EXP01_Result containing collision statistics for the dimension
        """
        print(f"Testing {dimension}D coordinate space...")

        # Calculate theoretical coordinate space
        coord_space_size = self.analyzer.calculate_coordinate_space_size(dimension)
        print(f"  Coordinate space: {coord_space_size:,} possible combinations")

        # Generate uniform coordinate samples and track collisions
        coordinates: Set[Tuple[int, ...]] = set()
        collisions = 0

        # Progress tracking for large sample sizes
        progress_interval = max(1, self.sample_size // 10)
        
        for i in range(self.sample_size):
            coord = self.analyzer.generate_coordinate(dimension, i)
            if coord in coordinates:
                collisions += 1
            coordinates.add(coord)

            # Show progress for large sample sizes
            if (i + 1) % progress_interval == 0:
                progress = (i + 1) / self.sample_size * 100
                print(f"  Progress: {progress:.1f}% ({i + 1:,}/{self.sample_size:,})")

        unique_coords = len(coordinates)
        collision_rate = collisions / self.sample_size if self.sample_size > 0 else 0.0
        geometric_limit_hit = self.sample_size > coord_space_size

        result = EXP01_Result(
            dimension=dimension,
            coordinate_space_size=coord_space_size,
            sample_size=self.sample_size,
            unique_coordinates=unique_coords,
            collisions=collisions,
            collision_rate=collision_rate,
            geometric_limit_hit=geometric_limit_hit,
        )

        # Display results for this dimension
        self._display_dimension_results(result)

        return result

    def _display_dimension_results(self, result: EXP01_Result) -> None:
        """Display formatted results for a single dimension test."""
        
        # Status based on geometric collision resistance pattern
        if result.dimension >= 4 and result.collision_rate < 0.001:  # Success: geometric collision resistance
            status = "GEOMETRICALLY RESISTANT"
            symbol = "PASS"
        elif result.dimension < 4 and result.collisions > 0:  # Expected: birthday paradox in smaller spaces
            status = "BIRTHDAY PARADOX (expected)"
            symbol = "CONFIRMED"
        elif result.dimension >= 4 and result.collision_rate >= 0.001:  # Fail: insufficient geometric resistance
            status = "WEAK COLLISION RESISTANCE"
            symbol = "FAIL"
        else:  # Low-D with no collisions (rare, but possible with small samples)
            status = "SAMPLE SPACE INSUFFICIENT"
            symbol = "WARNING"

        print(f"  {symbol} | Unique: {result.unique_coordinates:,} | Collisions: {result.collisions}")
        print(
            f"      Rate: {result.collision_rate * 100:.4f}% | Space: {'exceeded' if result.geometric_limit_hit else 'sufficient'}"
        )
        print(f"      Status: {status}")
        print()

    def run_comprehensive_test(self, dimensions: Optional[List[int]] = None) -> List[EXP01_Result]:
        """
        Run collision resistance tests across multiple dimensions.
        
        Args:
            dimensions: List of dimensions to test. If None, tests all supported dimensions.
            
        Returns:
            List of EXP01_Result objects for all tested dimensions
        """
        if dimensions is None:
            dimensions = self.analyzer.get_supported_dimensions()

        print(f"\n{'=' * 80}")
        print("EXP-01: GEOMETRIC COLLISION RESISTANCE TEST")
        print(f"{'=' * 80}")
        print(f"Sample size per dimension: {self.sample_size:,} coordinates")
        print(f"Testing dimensions: {', '.join(f'{d}D' for d in dimensions)}")
        print()

        results = []
        start_time = datetime.now()

        for dimension in dimensions:
            try:
                result = self.test_dimension(dimension)
                results.append(result)
            except Exception as e:
                print(f"  [ERROR] Failed to test {dimension}D: {e}")
                continue

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"{'=' * 80}")
        print(f"COMPREHENSIVE TEST COMPLETE")
        print(f"Duration: {duration}")
        print(f"Dimensions tested: {len(results)}")
        print(f"Total collisions detected: {sum(r.collisions for r in results)}")
        print(f"{'=' * 80}")

        return results

    def validate_geometric_resistance(self, results: List[EXP01_Result]) -> Dict[str, Any]:
        """
        Validate geometric collision resistance patterns across results.
        
        Args:
            results: List of collision test results across dimensions
            
        Returns:
            Dictionary containing geometric validation analysis
        """
        low_dim_results = [r for r in results if r.dimension < 4]
        high_dim_results = [r for r in results if r.dimension >= 4]

        if not low_dim_results or not high_dim_results:
            return {
                "error": "Insufficient dimension coverage for geometric validation",
                "low_dimensions": len(low_dim_results),
                "high_dimensions": len(high_dim_results)
            }

        # Calculate geometric validation metrics
        low_dim_collision_rate = sum(r.collision_rate for r in low_dim_results) / len(low_dim_results)
        high_dim_collision_rate = sum(r.collision_rate for r in high_dim_results) / len(high_dim_results)
        geometric_improvement = low_dim_collision_rate / high_dim_collision_rate if high_dim_collision_rate > 0 else float('inf')

        # Success criteria: High dimensions must show dramatically lower collision rates
        geometric_threshold_met = high_dim_collision_rate * 100 < low_dim_collision_rate  # 100x+ improvement

        return {
            "geometric_validation": {
                "low_dimensions_collisions": sum(r.collisions for r in low_dim_results),
                "low_dimensions_avg_collision_rate": low_dim_collision_rate,
                "high_dimensions_collisions": sum(r.collisions for r in high_dim_results),
                "high_dimensions_avg_collision_rate": high_dim_collision_rate,
                "geometric_improvement_factor": geometric_improvement,
                "geometric_transition_confirmed": geometric_threshold_met,
            },
            "validation_summary": {
                "low_dim_collision_rate_percent": low_dim_collision_rate * 100,
                "high_dim_collision_rate_percent": high_dim_collision_rate * 100,
                "improvement_factor": geometric_improvement,
                "validation_passed": geometric_threshold_met,
            }
        }

    def get_theoretical_analysis(self, dimensions: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get theoretical collision probability analysis for dimensions.
        
        Args:
            dimensions: List of dimensions to analyze. If None, analyzes all supported dimensions.
            
        Returns:
            Dictionary containing theoretical collision probability analysis
        """
        if dimensions is None:
            dimensions = self.analyzer.get_supported_dimensions()

        theoretical_analysis = {}
        
        for dimension in dimensions:
            analysis = self.analyzer.analyze_collision_probability(dimension, self.sample_size)
            theoretical_analysis[dimension] = analysis

        return theoretical_analysis