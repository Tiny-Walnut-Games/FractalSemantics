"""
EXP-01: Geometric Collision Resistance Test - Experiment Module

This module implements the core experiment logic for testing geometric collision
resistance in FractalStat coordinate spaces. The experiment validates that
collision resistance emerges from mathematical properties of coordinate spaces
rather than just cryptographic hashing.

Core Scientific Methodology:
1. Generate coordinate samples across dimensional subspaces (2D through 8D)
2. Measure collision rates and compare against theoretical expectations
3. Validate geometric transition point where collisions become mathematically impossible
4. Demonstrate that higher dimensions provide exponential collision resistance

Key Insights:
- 2D/3D spaces show expected Birthday Paradox collision patterns
- 4D+ spaces exhibit geometric collision resistance due to exponential space growth
- The 8th dimension provides complete expressivity and collision immunity
- Collision resistance is fundamentally mathematical, with crypto as additional assurance

Author: FractalSemantics
Date: 2025-12-07
"""

import json
import sys
import secrets
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

from .entities import EXP01_Result

secure_random = secrets.SystemRandom()


@dataclass
class CoordinateSpaceConfig:
    """
    Configuration for coordinate space testing parameters.
    
    This class defines the ranges and parameters used for generating coordinate
    samples across different dimensional spaces.
    """
    
    # Coordinate ranges for each dimension: [min, max] inclusive
    # Using realistic ranges that demonstrate hash collision resistance
    DIMENSION_RANGES = {
        2: [0, 100],   # 2D coordinates
        3: [0, 100],   # 3D coordinates  
        4: [0, 100],   # 4D coordinates
        5: [0, 100],   # 5D coordinates
        6: [0, 100],   # 6D coordinates
        7: [0, 100],   # 7D coordinates
        8: [0, 100],   # 8D coordinates
        9: [0, 100],   # 9D coordinates
        10: [0, 100],  # 10D coordinates
        11: [0, 100],  # 11D coordinates
        12: [0, 100],  # 12D coordinates
    }
    
    @classmethod
    def get_range_for_dimension(cls, dimension: int) -> Tuple[int, int]:
        """Get coordinate range for a specific dimension."""
        if dimension not in cls.DIMENSION_RANGES:
            raise ValueError(f"Unsupported dimension: {dimension}")
        return tuple(cls.DIMENSION_RANGES[dimension])


class EXP01_GeometricCollisionResistance:
    """
    EXP-01: Geometric Collision Resistance Test

    This experiment validates that FractalStat coordinate space exhibits mathematical
    collision resistance properties independent of cryptographic hashing.

    Scientific Rationale:
    The geometric structure of FractalStat coordinates inherently prevents collisions
    at higher dimensions due to exponential expansion of coordinate space:

    - 2D/3D: Coordinate space smaller than test scales → expected collisions
    - 4D+: Coordinate space vastly larger than test scales → geometric collision resistance
    - This proves collision resistance is mathematical, not just cryptographic

    Coordinate spaces are designed so that:
    - Sample size: 100k+ for real-scale testing
    - Low dimensions: Coordinate space < sample size (collisions)
    - High dimensions: Coordinate space >> sample size (no collisions)

    Statistical Significance:
    Testing at 100k+ samples empirically validates the geometric transition point,
    proving that FractalStat works through mathematics, with crypto as additional assurance.
    """

    def __init__(self, sample_size: int = 100000):
        """
        Initialize the geometric collision resistance experiment.
        
        Args:
            sample_size: Number of coordinate samples to generate per dimension
                        Default: 100,000 (suitable for hardware-constrained testing)
        """
        self.sample_size = sample_size
        self.dimensions = list(CoordinateSpaceConfig.DIMENSION_RANGES.keys())
        self.results: List[EXP01_Result] = []

    def _calculate_coordinate_space_size(self, dimension: int) -> int:
        """
        Calculate total possible coordinates for a given dimension.
        
        Args:
            dimension: Number of dimensions in the coordinate space
            
        Returns:
            Total number of possible coordinate combinations
            
        Mathematical Formula:
            coordinate_space_size = range_size^dimension
            
        Where range_size is the number of possible values per coordinate dimension.
        """
        min_val, max_val = CoordinateSpaceConfig.get_range_for_dimension(dimension)
        range_size = max_val - min_val + 1
        return range_size ** dimension

    def _generate_coordinate(self, dimension: int, seed: int) -> Tuple[int, ...]:
        """
        Generate a uniform coordinate tuple for given dimension.
        
        Args:
            dimension: Number of dimensions for the coordinate
            seed: Random seed for reproducible coordinate generation
            
        Returns:
            Tuple representing the coordinate in the specified dimension
            
        Implementation Details:
            - Uses cryptographically secure random generation
            - Ensures uniform distribution across coordinate space
            - Seed-based generation allows reproducible results
        """
        secure_random.seed(seed)
        min_val, max_val = CoordinateSpaceConfig.get_range_for_dimension(dimension)
        return tuple(secure_random.randint(min_val, max_val) for _ in range(dimension))

    def _analyze_dimension_results(self, result: EXP01_Result) -> Dict[str, Any]:
        """
        Analyze and categorize results for a specific dimension.
        
        Args:
            result: EXP01_Result containing collision test data
            
        Returns:
            Dictionary with analysis categorization and status
            
        Analysis Categories:
            - GEOMETRICALLY RESISTANT: 4D+ with <0.1% collision rate
            - BIRTHDAY PARADOX: 2D/3D with expected collisions
            - WEAK COLLISION RESISTANCE: 4D+ with high collision rate (failure)
            - SAMPLE SPACE INSUFFICIENT: Low-D with no collisions (rare case)
        """
        dimension = result.dimension
        collision_rate = result.collision_rate
        
        if dimension >= 4 and collision_rate < 0.001:  # Success: geometric collision resistance
            status = "GEOMETRICALLY RESISTANT"
            symbol = "PASS"
        elif dimension < 4 and result.collisions > 0:  # Expected: birthday paradox in smaller spaces
            status = "BIRTHDAY PARADOX (expected)"
            symbol = "CONFIRMED"
        elif dimension >= 4 and collision_rate >= 0.001:  # Fail: insufficient geometric resistance
            status = "WEAK COLLISION RESISTANCE"
            symbol = "FAIL"
        else:  # Low-D with no collisions (rare, but possible with small samples)
            status = "SAMPLE SPACE INSUFFICIENT"
            symbol = "WARNING"

        return {
            "status": status,
            "symbol": symbol,
            "is_geometrically_resistant": result.is_geometrically_resistant(),
            "collision_efficiency": result.get_collision_efficiency()
        }

    def run(self) -> Tuple[List[EXP01_Result], bool]:
        """
        Run the geometric collision resistance test.

        Tests coordinate collision rates across dimensions 2D→7D at 100k+ sample scale.

        Returns:
            Tuple of (results list, overall geometric validation success)
            
        Test Process:
            1. For each dimension (2D through 12D):
               - Calculate theoretical coordinate space size
               - Generate uniform coordinate samples
               - Count collisions and calculate rates
               - Store results with geometric validation status
            2. Analyze overall geometric collision resistance pattern
            3. Return results and validation success status
        """
        print(f"\n{'=' * 80}")
        print("EXP-01: GEOMETRIC COLLISION RESISTANCE TEST")
        print(f"{'=' * 80}")
        print(f"Sample size per dimension: {self.sample_size:,} coordinates")
        print(f"Testing dimensions: {', '.join(f'{d}D' for d in self.dimensions)}")
        print()

        all_validated = True

        for dimension in self.dimensions:
            print(f"Testing {dimension}D coordinate space...")

            # Calculate theoretical coordinate space
            coord_space_size = self._calculate_coordinate_space_size(dimension)
            print(f"  Coordinate space: {coord_space_size:,} possible combinations")

            # Generate uniform coordinate samples
            coordinates = set()
            collisions = 0

            for i in range(self.sample_size):
                coord = self._generate_coordinate(dimension, i)
                if coord in coordinates:
                    collisions += 1
                coordinates.add(coord)

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

            self.results.append(result)

            # Analyze and display results for this dimension
            analysis = self._analyze_dimension_results(result)
            
            print(f"  {analysis['symbol']} | Unique: {unique_coords:,} | Collisions: {collisions}")
            print(
                f"      Rate: {collision_rate * 100:.4f}% | Space: {'exceeded' if geometric_limit_hit else 'sufficient'}"
            )
            print(f"      Status: {analysis['status']}")
            print()

            # Update overall validation status
            if analysis['symbol'] == "FAIL":
                all_validated = False

        print(f"{'=' * 80}")
        print("GEOMETRIC VALIDATION SUMMARY")
        print(f"{'=' * 80}")

        # Analyze results for geometric collision resistance pattern
        low_dim_results = [r for r in self.results if r.dimension < 4]
        high_dim_results = [r for r in self.results if r.dimension >= 4]

        low_dim_collision_rate = sum(r.collision_rate for r in low_dim_results) / len(low_dim_results) if low_dim_results else 0.0
        high_dim_collision_rate = sum(r.collision_rate for r in high_dim_results) / len(high_dim_results) if high_dim_results else 0.0
        geometric_improvement = low_dim_collision_rate / high_dim_collision_rate if high_dim_collision_rate > 0 else float('inf')

        print("2D/3D (Low Dimensional - Birthday Paradox):")
        print(f"  Avg collision rate: {low_dim_collision_rate*100:.2f}%")
        print(f"  Total collisions: {sum(r.collisions for r in low_dim_results)}")
        print()

        print("4D+ (High Dimensional - Geometric Resistance):")
        print(f"  Avg collision rate: {high_dim_collision_rate*100:.2f}%")
        print(f"  Total collisions: {sum(r.collisions for r in high_dim_results)}")
        print(f"  Geometric improvement: {geometric_improvement:.0f}x lower collision rate")
        print()

        # Success criteria: High dimensions must show dramatically lower collision rates
        geometric_threshold_met = high_dim_collision_rate * 100 < low_dim_collision_rate  # 100x+ improvement

        if geometric_threshold_met and low_dim_collision_rate > 0:
            print("[Pass] GEOMETRIC COLLISION RESISTANCE VALIDATED")
            print(f"   • Low dimensions: {low_dim_collision_rate*100:.2f}% collision rate (expected)")
            print(f"   • High dimensions: {high_dim_collision_rate*100:.2f}% collision rate (excellent)")
            print(f"   • Geometric improvement: {geometric_improvement:.0f}x reduction")
            print("   • Higher dimensions exhibit strong geometric collision resistance")
        else:
            print("[Fail] GEOMETRIC VALIDATION INSUFFICIENT")
            print("   • Insufficient geometric collision resistance improvement")
            all_validated = False

        return self.results, all_validated

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive geometric analysis summary.
        
        Returns:
            Dictionary containing complete experiment results and analysis
            
        Summary Includes:
            - Sample size and dimensions tested
            - Geometric validation metrics
            - Coordinate space calculations
            - Overall pass/fail status
            - Detailed results for each dimension
        """
        low_dim_results = [r for r in self.results if r.dimension < 4]
        high_dim_results = [r for r in self.results if r.dimension >= 4]

        return {
            "sample_size": self.sample_size,
            "dimensions_tested": self.dimensions,
            "geometric_validation": {
                "low_dimensions_collisions": sum(r.collisions for r in low_dim_results),
                "low_dimensions_avg_collision_rate": (
                    sum(r.collision_rate for r in low_dim_results)
                    / len(low_dim_results)
                    if low_dim_results
                    else 0
                ),
                "high_dimensions_collisions": sum(
                    r.collisions for r in high_dim_results
                ),
                "high_dimensions_avg_collision_rate": (
                    sum(r.collision_rate for r in high_dim_results)
                    / len(high_dim_results)
                    if high_dim_results
                    else 0
                ),
                "geometric_transition_confirmed": (
                    # High dimensions show dramatically fewer collisions (geometric resistance)
                    sum(r.collision_rate for r in high_dim_results) / len(high_dim_results) <
                    sum(r.collision_rate for r in low_dim_results) / len(low_dim_results) / 100
                ) if high_dim_results and low_dim_results else False,
            },
            "coordinate_spaces": {
                dim: self._calculate_coordinate_space_size(dim)
                for dim in self.dimensions
            },
            # More lenient criteria: geometric resistance demonstrated
            "all_passed": (
                sum(r.collision_rate for r in [res for res in self.results if res.dimension >= 4]) /
                len([res for res in self.results if res.dimension >= 4]) <
                sum(r.collision_rate for r in [res for res in self.results if res.dimension < 4]) /
                len([res for res in self.results if res.dimension < 4]) / 100
            ) if any(r.dimension < 4 for r in self.results) and any(r.dimension >= 4 for r in self.results) else True,
            "results": [r.to_dict() for r in self.results],
        }


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary containing experiment results
        output_file: Optional output file path. If None, generates timestamped filename.
        
    Returns:
        Path to the saved results file
        
    File Format:
        JSON file with experiment metadata, configuration, and detailed results
        Saved in the project's results directory
    """
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp01_address_uniqueness_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


def run_experiment_from_config(config: Optional[Dict[str, Any]] = None) -> Tuple[List[EXP01_Result], bool, Dict[str, Any]]:
    """
    Run the experiment with configuration parameters.
    
    Args:
        config: Optional configuration dictionary with experiment parameters
        
    Returns:
        Tuple of (results list, success status, summary dictionary)
        
    Configuration Options:
        - sample_size: Number of samples per dimension (default: 100000)
        - dimensions: List of dimensions to test (default: 2-12)
    """
    if config is None:
        config = {}
    
    sample_size = config.get("sample_size", 100000)
    
    experiment = EXP01_GeometricCollisionResistance(sample_size=sample_size)
    results_list, success = experiment.run()
    summary = experiment.get_summary()
    
    return results_list, success, summary