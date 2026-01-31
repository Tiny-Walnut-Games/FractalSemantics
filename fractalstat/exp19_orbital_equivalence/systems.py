"""
EXP-19: Orbital Equivalence - Known Orbital Systems

Provides predefined orbital systems in both classical and fractal representations
for testing the equivalence hypothesis. Includes Earth-Sun system, Solar System,
and other celestial configurations.

Key Systems:
- Earth-Sun: Two-body system for fundamental testing
- Solar System: Multi-body system with inner planets
- Binary Stars: Stellar systems for different mass ratios
- Exoplanetary Systems: For testing with different orbital parameters

Scientific Foundation:
These systems provide controlled test cases where classical orbital mechanics
predictions are well-established, allowing rigorous validation of whether
fractal mechanics produces identical results.
"""

from typing import Tuple, List, Dict, Any
from .entities import CelestialBody, OrbitalSystem, FractalBody, FractalOrbitalSystem


def create_earth_sun_system() -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """Create Earth-Sun system in both classical and fractal representations.

    This is the fundamental test case for orbital equivalence:
    - Two-body system with well-understood classical dynamics
    - Provides reference for calibrating fractal parameters
    - Tests basic gravitational vs fractal cohesion equivalence

    Returns:
        Tuple of (classical_system, fractal_system)
    """
    # Classical representation
    sun = CelestialBody(
        name="Sun",
        mass=1.989e30,  # kg
        radius=6.96e8,   # m
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )

    earth = CelestialBody(
        name="Earth",
        mass=5.972e24,  # kg
        radius=6.371e6,  # m
        position=np.array([1.496e11, 0.0, 0.0]),  # 1 AU
        velocity=np.array([0.0, 2.978e4, 0.0])    # ~30 km/s orbital velocity
    )

    classical_system = OrbitalSystem(
        name="Earth-Sun",
        bodies=[sun, earth],
        central_body=sun
    )

    # Fractal representation
    fractal_sun = FractalBody(
        name="Sun",
        fractal_density=1.0,  # Reference density
        hierarchical_depth=0,  # Central body
        tree_address=[]
    )

    fractal_earth = FractalBody(
        name="Earth",
        fractal_density=0.000003,  # Earth/Sun mass ratio ≈ 3e-6
        hierarchical_depth=1,       # Inner planet
        tree_address=[0]
    )

    fractal_system = FractalOrbitalSystem(
        name="Earth-Sun",
        bodies=[fractal_sun, fractal_earth],
        central_body=fractal_sun,
        max_hierarchy_depth=2,
        cohesion_constant=1.0  # Will be calibrated to match gravity
    )

    return classical_system, fractal_system


def create_solar_system() -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """Create simplified Solar System in both representations.

    Includes inner planets for multi-body testing:
    - Tests gravitational interactions between multiple bodies
    - Validates fractal hierarchy mapping for different orbital distances
    - Provides more complex test case than Earth-Sun system

    Returns:
        Tuple of (classical_system, fractal_system)
    """
    import numpy as np

    # Classical representation (inner planets only for simplicity)
    sun = CelestialBody("Sun", 1.989e30, 6.96e8, np.array([0., 0., 0.]), np.array([0., 0., 0.]))

    bodies = [sun]
    planets_data = [
        ("Mercury", 3.301e23, 2.440e6, 0.387, 4.74e4),
        ("Venus", 4.867e24, 6.052e6, 0.723, 3.50e4),
        ("Earth", 5.972e24, 6.371e6, 1.000, 2.98e4),
        ("Mars", 6.39e23, 3.390e6, 1.524, 2.41e4),
    ]

    for name, mass, radius, au_distance, velocity in planets_data:
        distance_m = au_distance * 1.496e11  # AU to meters
        pos = np.array([distance_m, 0., 0.])
        vel = np.array([0., velocity, 0.])
        bodies.append(CelestialBody(name, mass, radius, pos, vel))

    classical_system = OrbitalSystem("Solar System", bodies, sun)

    # Fractal representation
    fractal_sun = FractalBody("Sun", 1.0, 0, [])

    fractal_bodies = [fractal_sun]
    fractal_densities = [0.000166, 0.00000245, 0.000003, 0.00000032]  # Mass ratios

    for i, (name, _, _, _, _) in enumerate(planets_data):
        fractal_bodies.append(FractalBody(
            name=name,
            fractal_density=fractal_densities[i],
            hierarchical_depth=i+1,
            tree_address=[i]
        ))

    fractal_system = FractalOrbitalSystem(
        "Solar System",
        fractal_bodies,
        fractal_sun,
        max_hierarchy_depth=5,
        cohesion_constant=1.0
    )

    return classical_system, fractal_system


def create_binary_star_system() -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """Create binary star system for testing different mass ratios.

    Binary systems test:
    - Systems without a single dominant central body
    - Different mass ratios and orbital configurations
    - Barycenter dynamics in both frameworks

    Returns:
        Tuple of (classical_system, fractal_system)
    """
    import numpy as np

    # Classical representation: Two stars of different masses
    star1 = CelestialBody(
        name="Star A",
        mass=2.0e30,  # Slightly more massive than Sun
        radius=7.0e8,
        position=np.array([-1.0e11, 0.0, 0.0]),  # Offset from center
        velocity=np.array([0.0, 1.5e4, 0.0])     # Orbital velocity
    )

    star2 = CelestialBody(
        name="Star B",
        mass=1.5e30,  # Less massive companion
        radius=6.0e8,
        position=np.array([1.33e11, 0.0, 0.0]),   # Different distance
        velocity=np.array([0.0, -2.0e4, 0.0])    # Opposite velocity
    )

    classical_system = OrbitalSystem(
        name="Binary Stars",
        bodies=[star1, star2],
        central_body=star1  # Conventionally the more massive
    )

    # Fractal representation
    fractal_star1 = FractalBody(
        name="Star A",
        fractal_density=1.0,      # Reference density
        hierarchical_depth=0,     # Primary in hierarchy
        tree_address=[]
    )

    fractal_star2 = FractalBody(
        name="Star B",
        fractal_density=0.75,     # 75% of primary density
        hierarchical_depth=1,     # Companion in hierarchy
        tree_address=[0]
    )

    fractal_system = FractalOrbitalSystem(
        name="Binary Stars",
        bodies=[fractal_star1, fractal_star2],
        central_body=fractal_star1,
        max_hierarchy_depth=2,
        cohesion_constant=1.0
    )

    return classical_system, fractal_system


def create_exoplanetary_system() -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """Create exoplanetary system with different orbital parameters.

    Tests the framework with:
    - Different stellar masses and planetary configurations
    - Non-solar system parameters
    - Validation that the equivalence works beyond our Solar System

    Returns:
        Tuple of (classical_system, fractal_system)
    """
    import numpy as np

    # Classical representation: Hot Jupiter system
    host_star = CelestialBody(
        name="Host Star",
        mass=1.2e30,  # Slightly more massive than Sun
        radius=7.5e8,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )

    hot_jupiter = CelestialBody(
        name="Hot Jupiter",
        mass=1.9e27,   # Jupiter mass
        radius=7.0e7,  # Large radius due to heating
        position=np.array([7.0e9, 0.0, 0.0]),   # Very close orbit (0.05 AU)
        velocity=np.array([0.0, 1.5e5, 0.0])    # Very fast orbital velocity
    )

    classical_system = OrbitalSystem(
        name="Exoplanetary System",
        bodies=[host_star, hot_jupiter],
        central_body=host_star
    )

    # Fractal representation
    fractal_host = FractalBody(
        name="Host Star",
        fractal_density=1.0,
        hierarchical_depth=0,
        tree_address=[]
    )

    fractal_jupiter = FractalBody(
        name="Hot Jupiter",
        fractal_density=0.00095,  # Jupiter/Sun mass ratio
        hierarchical_depth=1,     # Close orbit
        tree_address=[0]
    )

    fractal_system = FractalOrbitalSystem(
        name="Exoplanetary System",
        bodies=[fractal_host, fractal_jupiter],
        central_body=fractal_host,
        max_hierarchy_depth=2,
        cohesion_constant=1.0
    )

    return classical_system, fractal_system


def create_lunar_system() -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """Create Earth-Moon system for satellite testing.

    Tests:
    - Satellite dynamics in both frameworks
    - Three-body system (Sun-Earth-Moon) interactions
    - Different scale of gravitational interactions

    Returns:
        Tuple of (classical_system, fractal_system)
    """
    import numpy as np

    # Classical representation
    earth = CelestialBody(
        name="Earth",
        mass=5.972e24,
        radius=6.371e6,
        position=np.array([1.496e11, 0.0, 0.0]),
        velocity=np.array([0.0, 2.978e4, 0.0])
    )

    moon = CelestialBody(
        name="Moon",
        mass=7.342e22,
        radius=1.737e6,
        position=np.array([1.496e11 + 3.844e8, 0.0, 0.0]),  # Earth-Moon distance
        velocity=np.array([0.0, 2.978e4 + 1.022e3, 0.0])    # Earth velocity + orbital velocity
    )

    classical_system = OrbitalSystem(
        name="Earth-Moon",
        bodies=[earth, moon],
        central_body=earth
    )

    # Fractal representation
    fractal_earth = FractalBody(
        name="Earth",
        fractal_density=0.000003,
        hierarchical_depth=1,
        tree_address=[0]
    )

    fractal_moon = FractalBody(
        name="Moon",
        fractal_density=0.0000000037,  # Moon/Earth mass ratio
        hierarchical_depth=2,          # Satellite hierarchy
        tree_address=[0, 0]
    )

    fractal_system = FractalOrbitalSystem(
        name="Earth-Moon",
        bodies=[fractal_earth, fractal_moon],
        central_body=fractal_earth,
        max_hierarchy_depth=3,
        cohesion_constant=1.0
    )

    return classical_system, fractal_system


def get_all_test_systems() -> Dict[str, Tuple[OrbitalSystem, FractalOrbitalSystem]]:
    """Get all predefined test systems.

    Returns:
        Dictionary mapping system names to (classical, fractal) system pairs
    """
    return {
        "Earth-Sun": create_earth_sun_system(),
        "Solar System": create_solar_system(),
        "Binary Stars": create_binary_star_system(),
        "Exoplanetary System": create_exoplanetary_system(),
        "Earth-Moon": create_lunar_system(),
    }


def validate_system_mapping(
    classical_system: OrbitalSystem,
    fractal_system: FractalOrbitalSystem
) -> Dict[str, Any]:
    """Validate that classical and fractal systems are properly mapped.

    Checks:
    - All classical bodies have corresponding fractal bodies
    - Mass ratios are preserved in fractal densities
    - Orbital distance mappings are reasonable
    - System topology is consistent

    Args:
        classical_system: Classical orbital system
        fractal_system: Corresponding fractal system

    Returns:
        Validation results with any issues found
    """
    validation_results = {
        'body_mapping_valid': True,
        'mass_ratio_preserved': True,
        'distance_mapping_valid': True,
        'issues': []
    }

    # Check body mapping
    classical_names = {body.name for body in classical_system.bodies}
    fractal_names = {body.name for body in fractal_system.bodies}

    if classical_names != fractal_names:
        validation_results['body_mapping_valid'] = False
        missing = classical_names - fractal_names
        extra = fractal_names - classical_names
        if missing:
            validation_results['issues'].append(f"Missing fractal bodies: {missing}")
        if extra:
            validation_results['issues'].append(f"Extra fractal bodies: {extra}")

    # Check mass ratio preservation (for systems with multiple bodies)
    if len(classical_system.bodies) > 1:
        central_classical = classical_system.central_body
        central_fractal = fractal_system.central_body

        for classical_body in classical_system.bodies:
            if classical_body is not central_classical:
                fractal_body = next((b for b in fractal_system.bodies if b.name == classical_body.name), None)
                if fractal_body:
                    # Calculate mass ratios
                    classical_ratio = classical_body.mass / central_classical.mass
                    fractal_ratio = fractal_body.fractal_density / central_fractal.fractal_density

                    # Check if ratios are reasonably preserved (within factor of 10)
                    ratio_diff = abs(classical_ratio - fractal_ratio) / classical_ratio if classical_ratio > 0 else 0
                    if ratio_diff > 0.5:  # 50% difference threshold
                        validation_results['mass_ratio_preserved'] = False
                        validation_results['issues'].append(
                            f"Mass ratio mismatch for {classical_body.name}: "
                            f"classical={classical_ratio:.2e}, fractal={fractal_ratio:.2e}"
                        )

    # Check distance mapping
    for classical_body in classical_system.bodies:
        if classical_body is not classical_system.central_body:
            fractal_body = next((b for b in fractal_system.bodies if b.name == classical_body.name), None)
            if fractal_body:
                # Check if hierarchical depth reasonably maps to distance
                classical_distance = np.linalg.norm(classical_body.position)
                fractal_distance = fractal_body.orbital_distance * 1.496e11  # Convert AU to meters

                distance_ratio = classical_distance / fractal_distance if fractal_distance > 0 else float('inf')
                if not (0.1 < distance_ratio < 10):  # Orders of magnitude check
                    validation_results['distance_mapping_valid'] = False
                    validation_results['issues'].append(
                        f"Distance mapping issue for {classical_body.name}: "
                        f"classical={classical_distance:.2e}m, fractal={fractal_distance:.2e}m"
                    )

    return validation_results


def create_custom_system(
    classical_bodies: List[CelestialBody],
    fractal_bodies: List[FractalBody],
    system_name: str = "Custom System"
) -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """Create a custom orbital system from provided bodies.

    Allows users to define their own test cases for orbital equivalence testing.

    Args:
        classical_bodies: List of classical celestial bodies
        fractal_bodies: List of corresponding fractal bodies
        system_name: Name for the custom system

    Returns:
        Tuple of (classical_system, fractal_system)

    Raises:
        ValueError: If body lists don't match or are invalid
    """
    if not classical_bodies or not fractal_bodies:
        raise ValueError("Both classical and fractal body lists must be non-empty")

    if len(classical_bodies) != len(fractal_bodies):
        raise ValueError("Classical and fractal body lists must have the same length")

    # Find most massive classical body as central
    central_classical = max(classical_bodies, key=lambda b: b.mass)

    classical_system = OrbitalSystem(
        name=system_name,
        bodies=classical_bodies,
        central_body=central_classical
    )

    # Find highest density fractal body as central
    central_fractal = max(fractal_bodies, key=lambda b: b.fractal_density)

    max_depth = max(body.hierarchical_depth for body in fractal_bodies)

    fractal_system = FractalOrbitalSystem(
        name=system_name,
        bodies=fractal_bodies,
        central_body=central_fractal,
        max_hierarchy_depth=max_depth,
        cohesion_constant=1.0
    )

    return classical_system, fractal_system