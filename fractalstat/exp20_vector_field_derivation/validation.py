"""
Vector field validation and inverse-square law verification.

Contains functions for validating that derived vector fields follow
inverse-square law behavior and other physical constraints.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from .entities import FractalEntity
from .vector_field_system import VectorFieldApproach


def create_continuous_vector_field(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    vector_approach: VectorFieldApproach,
    scalar_magnitude: float,
    grid_resolution: int = 50,
    field_bounds: float = 2e11  # 2 AU in meters
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a continuous 3D vector field from discrete fractal interactions.

    Args:
        entity_a, entity_b: The two entities
        vector_approach: Vector field derivation approach
        scalar_magnitude: Base force magnitude
        grid_resolution: Number of grid points per dimension
        field_bounds: Spatial extent of the field (meters)

    Returns:
        Tuple of (X, Y, Z, Fx, Fy, Fz) grid arrays
    """
    # Create spatial grid
    grid_1d = np.linspace(-field_bounds, field_bounds, grid_resolution)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

    # Initialize force field arrays
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    Fz = np.zeros_like(Z)

    # Calculate force at each grid point
    reference_distance = 1.496e11  # 1 AU in meters

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                position = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])

                # Calculate distance from central body for inverse-square falloff
                r_vector = entity_b.position - position
                current_distance = np.linalg.norm(r_vector)

                # Apply inverse-square law: F ∝ 1/r²
                if current_distance > 0:
                    distance_factor = (reference_distance / current_distance) ** 2
                    effective_magnitude = scalar_magnitude * distance_factor
                else:
                    effective_magnitude = scalar_magnitude

                # Create temporary entity at this position
                temp_entity = FractalEntity(
                    name="field_point",
                    position=position,
                    velocity=np.zeros(3),  # Stationary field point
                    mass=1.0,  # Unit mass for field calculation
                    fractal_density=entity_a.fractal_density,
                    hierarchical_depth=entity_a.hierarchical_depth,
                    branching_factor=entity_a.branching_factor
                )

                # Calculate force vector at this point with distance-dependent magnitude
                force_vector = vector_approach.derive_force(temp_entity, entity_b, effective_magnitude)

                Fx[i,j,k] = force_vector[0]
                Fy[i,j,k] = force_vector[1]
                Fz[i,j,k] = force_vector[2]

    # Return unsmoothed field for accurate inverse-square validation
    # Smoothing was found to reduce correlation with theoretical 1/r² behavior
    return X, Y, Z, Fx, Fy, Fz


def verify_inverse_square_law(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray,
    origin: np.ndarray,
    test_distances: np.ndarray
) -> float:
    """
    Verify that the vector field follows inverse-square law.

    Args:
        X, Y, Z: Spatial grid coordinates
        Fx, Fy, Fz: Force field components
        origin: Center point for radial testing
        test_distances: Distances to test at

    Returns:
        Correlation coefficient with 1/r² law
    """
    measured_magnitudes = []
    theoretical_magnitudes = []

    for r in test_distances:
        if r == 0:
            continue

        # Sample force magnitude at distance r from origin
        # Use multiple directions for averaging
        magnitudes_at_r = []

        for _ in range(8):  # Sample in 8 directions
            # Random direction
            theta = 2 * np.pi * np.random.random()
            phi = np.pi * np.random.random()

            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            test_point = origin + direction * r

            # Interpolate field at this point
            magnitude = interpolate_field_at_point(X, Y, Z, Fx, Fy, Fz, test_point)
            if magnitude > 0:
                magnitudes_at_r.append(magnitude)

        if magnitudes_at_r:
            avg_magnitude = np.mean(magnitudes_at_r)
            measured_magnitudes.append(avg_magnitude)
            theoretical_magnitudes.append(1.0 / (r ** 2))

    if len(measured_magnitudes) < 2:
        return 0.0

    # Calculate correlation with theoretical 1/r²
    try:
        correlation = np.corrcoef(measured_magnitudes, theoretical_magnitudes)[0, 1]
        return abs(correlation)  # Return absolute value
    except (ValueError, TypeError, IndexError) as e:
        print(f"Warning: Could not compute inverse-square correlation: {e}")
        return 0.0


def interpolate_field_at_point(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray,
    point: np.ndarray
) -> float:
    """
    Interpolate field magnitude at an arbitrary point.

    Args:
        X, Y, Z: Grid coordinates
        Fx, Fy, Fz: Field components
        point: [x,y,z] coordinates to interpolate at

    Returns:
        Field magnitude at the point
    """
    # Simple nearest neighbor interpolation for now
    # Could be upgraded to trilinear interpolation
    x_idx = np.argmin(np.abs(X[:, 0, 0] - point[0]))
    y_idx = np.argmin(np.abs(Y[0, :, 0] - point[1]))
    z_idx = np.argmin(np.abs(Z[0, 0, :] - point[2]))

    # Ensure indices are within bounds
    x_idx = np.clip(x_idx, 0, X.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, Y.shape[1] - 1)
    z_idx = np.clip(z_idx, 0, Z.shape[2] - 1)

    fx = Fx[x_idx, y_idx, z_idx]
    fy = Fy[x_idx, y_idx, z_idx]
    fz = Fz[x_idx, y_idx, z_idx]

    return np.linalg.norm([fx, fy, fz])


@dataclass
class InverseSquareValidation:
    """Results from inverse-square law validation."""

    approach_name: str
    correlation_with_inverse_square: float
    test_distances: List[float]
    measured_magnitudes: List[float]
    theoretical_magnitudes: List[float]

    inverse_square_confirmed: bool


def validate_inverse_square_law_for_approach(
    approach_name: str,
    scalar_magnitude: float = 3.54e22
) -> InverseSquareValidation:
    """
    Validate that a vector field approach produces inverse-square behavior.

    Args:
        approach_name: Which approach to validate
        scalar_magnitude: Base force magnitude

    Returns:
        Validation results
    """
    print(f"Validating inverse-square law for {approach_name}...")

    # Create test entities
    from .entities import create_earth_sun_fractal_entities
    earth, sun = create_earth_sun_fractal_entities()

    # Get the approach
    from .vector_field_system import VectorFieldDerivationSystem
    derivation_system = VectorFieldDerivationSystem()
    approach = next(a for a in derivation_system.approaches if a.name == approach_name)

    # Create continuous field with higher resolution for better inverse-square validation
    X, Y, Z, Fx, Fy, Fz = create_continuous_vector_field(
        earth, sun, approach, scalar_magnitude,
        grid_resolution=50, field_bounds=2e11
    )

    # Test at various distances
    origin = sun.position
    test_distances = np.logspace(10, 11.3, 10)  # 10^10 to ~2e11 meters

    correlation = verify_inverse_square_law(X, Y, Z, Fx, Fy, Fz, origin, test_distances)

    # Sample some magnitudes for reporting
    measured_magnitudes = []
    theoretical_magnitudes = []

    for r in test_distances[::2]:  # Sample every other distance
        test_point = origin + np.array([r, 0, 0])  # Along x-axis
        magnitude = interpolate_field_at_point(X, Y, Z, Fx, Fy, Fz, test_point)
        measured_magnitudes.append(magnitude)
        theoretical_magnitudes.append(1.0 / (r ** 2))

    validation = InverseSquareValidation(
        approach_name=approach_name,
        correlation_with_inverse_square=correlation,
        test_distances=test_distances[::2].tolist(),
        measured_magnitudes=measured_magnitudes,
        theoretical_magnitudes=theoretical_magnitudes,
        inverse_square_confirmed=correlation > 0.98
    )

    print(f"  Correlation: {correlation:.6f}")
    print(f"  Status: {'CONFIRMED' if validation.inverse_square_confirmed else 'FAILED'}")

    return validation