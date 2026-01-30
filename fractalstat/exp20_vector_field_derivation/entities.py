"""
Entity definitions for vector field derivation.

Contains the FractalEntity class and related data structures for representing
celestial bodies with fractal properties in the vector field derivation system.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timezone


@dataclass
class FractalEntity:
    """Entity with fractal properties for vector field derivation."""

    name: str
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    mass: float  # kg
    fractal_density: float  # Fractal complexity measure
    hierarchical_depth: int  # Depth in fractal hierarchy
    branching_factor: int  # Branching complexity

    def __repr__(self):
        return f"Entity({self.name}, depth={self.hierarchical_depth}, branching={self.branching_factor})"


@dataclass
class VectorFieldResult:
    """Result of vector field derivation for a specific approach."""

    approach_name: str
    entity_a: FractalEntity
    entity_b: FractalEntity
    force_vector: np.ndarray  # [Fx, Fy, Fz] in Newtons
    magnitude_accuracy: float  # How well magnitude matches Newtonian
    direction_accuracy: float  # How well direction points correctly
    derivation_time: float  # Time taken to derive vector


def create_earth_sun_fractal_entities() -> Tuple[FractalEntity, FractalEntity]:
    """Create Earth-Sun system with fractal properties."""

    # Sun parameters (from EXP-14 and EXP-13)
    sun = FractalEntity(
        name="Sun",
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=1.989e30,  # kg
        fractal_density=1.0,  # Reference density
        hierarchical_depth=7,  # Deep hierarchy (stellar)
        branching_factor=25  # High branching (fusion processes)
    )

    # Earth parameters
    earth = FractalEntity(
        name="Earth",
        position=np.array([1.496e11, 0.0, 0.0]),  # 1 AU
        velocity=np.array([0.0, 2.978e4, 0.0]),   # ~30 km/s orbital velocity
        mass=5.972e24,  # kg
        fractal_density=0.333,  # Less complex than Sun
        hierarchical_depth=4,   # Planetary depth
        branching_factor=11    # Moderate branching (geological processes)
    )

    return earth, sun


def create_solar_system_fractal_entities() -> List[FractalEntity]:
    """Create multiple planets in fractal representation."""

    import secrets
    secure_random = secrets.SystemRandom()

    # Central Sun
    sun = FractalEntity(
        name="Sun",
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=1.989e30,
        fractal_density=1.0,
        hierarchical_depth=7,
        branching_factor=25
    )

    entities = [sun]

    # Planet data: (name, mass, distance_AU, velocity_km_s, depth, branching)
    planet_data = [
        ("Mercury", 3.301e23, 0.387, 47.4, 3, 8),
        ("Venus", 4.867e24, 0.723, 35.0, 3, 9),
        ("Earth", 5.972e24, 1.000, 29.8, 4, 11),
        ("Mars", 6.39e23, 1.524, 24.1, 4, 10),
        ("Jupiter", 1.898e27, 5.204, 13.1, 5, 16),
    ]

    for name, mass, au_dist, vel_km_s, depth, branching in planet_data:
        distance_m = au_dist * 1.496e11
        velocity_m_s = vel_km_s * 1000

        # Initialize in circular orbit
        angle = 2 * np.pi * secure_random.random()

        entity = FractalEntity(
            name=name,
            position=np.array([distance_m * np.cos(angle), distance_m * np.sin(angle), 0.0]),
            velocity=np.array([-velocity_m_s * np.sin(angle), velocity_m_s * np.cos(angle), 0.0]),
            mass=mass,
            fractal_density=0.1 + 0.1 * depth,  # Correlated with depth
            hierarchical_depth=depth,
            branching_factor=branching
        )
        entities.append(entity)

    return entities