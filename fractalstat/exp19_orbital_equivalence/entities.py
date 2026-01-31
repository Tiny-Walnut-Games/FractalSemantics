"""
EXP-19: Orbital Equivalence - Core Entities

Contains the fundamental data models for both classical Newtonian mechanics
and fractal cohesion mechanics representations of orbital systems.

This module defines the core entities used throughout the orbital equivalence testing:
- CelestialBody and OrbitalSystem for classical mechanics
- FractalBody and FractalOrbitalSystem for fractal mechanics

Scientific Rationale:
The equivalence test requires parallel representations of the same physical systems
in both classical and fractal frameworks. These entities provide the foundation
for comparing whether orbital mechanics and fractal cohesion mechanics produce
identical predictions for the same celestial configurations.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class CelestialBody:
    """Classical celestial body with Newtonian properties.

    Represents a celestial object using classical physics parameters:
    mass, radius, position, and velocity vectors.

    Attributes:
        name: Name of the celestial body (e.g., "Earth", "Sun")
        mass: Mass in kilograms (kg)
        radius: Radius in meters (m)
        position: Position vector [x, y, z] in meters
        velocity: Velocity vector [vx, vy, vz] in m/s
    """

    name: str
    mass: float  # kg
    radius: float  # m
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s

    def __repr__(self) -> str:
        """String representation showing key properties."""
        return f"Body({self.name}, m={self.mass:.2e}kg, pos={self.position})"

    def distance_to(self, other: 'CelestialBody') -> float:
        """Calculate distance to another celestial body."""
        return np.linalg.norm(other.position - self.position)

    def relative_velocity(self, other: 'CelestialBody') -> np.ndarray:
        """Calculate relative velocity with respect to another body."""
        return other.velocity - self.velocity


@dataclass
class OrbitalSystem:
    """Classical orbital system (e.g., Solar System, binary stars).

    Represents a collection of celestial bodies interacting through
    Newtonian gravitational forces.

    Attributes:
        name: Name of the orbital system
        bodies: List of celestial bodies in the system
        central_body: The primary body (usually most massive)
    """

    name: str
    bodies: List[CelestialBody]
    central_body: CelestialBody

    def __post_init__(self) -> None:
        """Validate system setup after initialization."""
        if self.central_body not in self.bodies:
            raise ValueError("Central body must be in bodies list")

    def get_body_by_name(self, name: str) -> Optional[CelestialBody]:
        """Get a body by its name."""
        for body in self.bodies:
            if body.name == name:
                return body
        return None

    def total_mass(self) -> float:
        """Calculate total mass of the system."""
        return sum(body.mass for body in self.bodies)

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass position."""
        total_mass = self.total_mass()
        if total_mass == 0:
            return np.zeros(3)

        weighted_positions = sum(body.mass * body.position for body in self.bodies)
        return weighted_positions / total_mass

    def kinetic_energy(self) -> float:
        """Calculate total kinetic energy of the system."""
        return sum(0.5 * body.mass * np.linalg.norm(body.velocity)**2 for body in self.bodies)

    def potential_energy(self, G: float = 6.67430e-11) -> float:
        """Calculate total gravitational potential energy."""
        total_potential = 0.0
        n_bodies = len(self.bodies)

        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                body1, body2 = self.bodies[i], self.bodies[j]
                distance = body1.distance_to(body2)
                if distance > 0:
                    potential = -G * body1.mass * body2.mass / distance
                    total_potential += potential

        return total_potential

    def total_energy(self, G: float = 6.67430e-11) -> float:
        """Calculate total mechanical energy (kinetic + potential)."""
        return self.kinetic_energy() + self.potential_energy(G)


@dataclass
class FractalBody:
    """Fractal representation of a celestial body.

    Represents a celestial object using fractal hierarchy parameters:
    fractal density (maps to mass) and hierarchical depth (maps to distance).

    Attributes:
        name: Name of the fractal body
        fractal_density: Density parameter that maps to effective mass
        hierarchical_depth: Depth in fractal hierarchy (maps to orbital distance)
        tree_address: Hierarchical position in the fractal tree structure
    """

    name: str
    fractal_density: float  # Maps to mass (higher density = more massive)
    hierarchical_depth: int  # Maps to orbital distance (deeper = farther)
    tree_address: List[int]  # Hierarchical position in fractal tree

    @property
    def orbital_distance(self) -> float:
        """Convert hierarchical depth to orbital distance in AU.

        Maps fractal hierarchy depth to astronomical units using
        a power-law relationship that approximates real orbital distances.
        """
        # Map hierarchical depth to orbital distance
        # Depth 1 = inner planets, Depth 4+ = outer planets/Kuiper belt
        return 0.4 * (self.hierarchical_depth ** 1.5)  # AU

    @property
    def effective_mass(self) -> float:
        """Convert fractal density to effective mass in solar masses.

        Maps fractal density parameter to stellar masses for comparison
        with classical mechanics.
        """
        # Map fractal density to stellar masses
        return self.fractal_density * 2.0  # Solar masses

    @property
    def fractal_radius(self) -> float:
        """Calculate effective radius based on fractal density."""
        # Simple scaling: higher density = larger effective radius
        return self.fractal_density * 0.1  # Arbitrary scaling factor

    def __repr__(self) -> str:
        """String representation showing fractal properties."""
        return (f"FractalBody({self.name}, density={self.fractal_density:.3f}, "
                f"depth={self.hierarchical_depth}, distance={self.orbital_distance:.2f}AU)")


@dataclass
class FractalOrbitalSystem:
    """Fractal representation of an orbital system.

    Represents a collection of fractal bodies interacting through
    fractal cohesion forces rather than classical gravity.

    Attributes:
        name: Name of the fractal orbital system
        bodies: List of fractal bodies in the system
        central_body: The primary fractal body (usually highest density)
        max_hierarchy_depth: Maximum depth in the fractal hierarchy
        cohesion_constant: Base strength of fractal cohesion forces
    """

    name: str
    bodies: List[FractalBody]
    central_body: FractalBody
    max_hierarchy_depth: int
    cohesion_constant: float  # Base cohesion strength

    def __post_init__(self) -> None:
        """Validate fractal system setup after initialization."""
        if self.central_body not in self.bodies:
            raise ValueError("Central body must be in bodies list")

    def get_body_by_name(self, name: str) -> Optional[FractalBody]:
        """Get a fractal body by its name."""
        for body in self.bodies:
            if body.name == name:
                return body
        return None

    def get_hierarchical_distance(self, body1: FractalBody, body2: FractalBody) -> float:
        """Calculate hierarchical distance between two fractal bodies.

        This maps to orbital distance in the fractal representation and
        determines the strength of fractal cohesion forces.

        Args:
            body1: First fractal body
            body2: Second fractal body

        Returns:
            Hierarchical distance (positive value)
        """
        # Simplified: difference in hierarchical depths
        depth_diff = abs(body1.hierarchical_depth - body2.hierarchical_depth)

        # Add branching distance if same depth
        if depth_diff == 0:
            # Calculate tree distance
            addr1 = body1.tree_address
            addr2 = body2.tree_address
            if len(addr1) == len(addr2):
                for i in range(len(addr1)):
                    if addr1[i] != addr2[i]:
                        depth_diff = 0.1 * (i + 1)  # Small separation for same depth
                        break

        return max(1.0, depth_diff)

    def total_fractal_density(self) -> float:
        """Calculate total fractal density of the system."""
        return sum(body.fractal_density for body in self.bodies)

    def average_hierarchical_depth(self) -> float:
        """Calculate average hierarchical depth."""
        if not self.bodies:
            return 0.0
        return sum(body.hierarchical_depth for body in self.bodies) / len(self.bodies)

    def cohesion_energy(self) -> float:
        """Calculate total fractal cohesion energy."""
        total_cohesion = 0.0
        n_bodies = len(self.bodies)

        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                body1, body2 = self.bodies[i], self.bodies[j]
                hierarchical_dist = self.get_hierarchical_distance(body1, body2)

                # Cohesion energy calculation
                density_product = body1.fractal_density * body2.fractal_density
                cohesion_energy = -self.cohesion_constant * density_product / hierarchical_dist
                total_cohesion += cohesion_energy

        return total_cohesion

    def effective_gravitational_constant(self) -> float:
        """Calculate effective G from fractal parameters.

        This is a key test: can we derive Newton's G from purely
        fractal topological parameters?
        """
        # For calibration: G should emerge from fractal parameters
        # This will be calculated during system calibration
        return self.cohesion_constant


def create_reference_system_mapping() -> Dict[str, Dict[str, Any]]:
    """Create reference mappings between classical and fractal parameters.

    Provides standard conversions for common celestial systems to
    facilitate comparison between frameworks.

    Returns:
        Dictionary mapping system names to parameter mappings
    """
    return {
        "Earth-Sun": {
            "classical": {
                "sun_mass": 1.989e30,  # kg
                "earth_mass": 5.972e24,  # kg
                "earth_distance": 1.496e11,  # meters (1 AU)
                "earth_velocity": 2.978e4,  # m/s
            },
            "fractal": {
                "sun_density": 1.0,  # Reference density
                "earth_density": 0.000003,  # Earth/Sun mass ratio
                "earth_depth": 1,  # Inner planet hierarchy
                "earth_address": [0],  # Tree position
            }
        },
        "Solar System": {
            "classical": {
                "planets": [
                    ("Mercury", 3.301e23, 0.387, 4.74e4),
                    ("Venus", 4.867e24, 0.723, 3.50e4),
                    ("Earth", 5.972e24, 1.000, 2.98e4),
                    ("Mars", 6.39e23, 1.524, 2.41e4),
                ]
            },
            "fractal": {
                "densities": [0.000166, 0.00000245, 0.000003, 0.00000032],
                "depths": [1, 2, 3, 4],
                "addresses": [[0], [1], [2], [3]],
            }
        }
    }