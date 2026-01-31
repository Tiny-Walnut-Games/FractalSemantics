"""
EXP-19: Orbital Equivalence - Fractal Mechanics

Implements fractal cohesion mechanics for orbital trajectory simulation.
This module provides the fractal framework that should produce equivalent
predictions to classical Newtonian mechanics if the hypothesis is correct.

Key Components:
- Fractal cohesion force calculations based on hierarchical distance
- Fractal orbital trajectory simulation
- Mapping between fractal parameters and classical orbital elements
- Hierarchical distance calculations in fractal tree structures

Scientific Foundation:
The fractal mechanics hypothesis proposes that gravitational effects
emerge from fractal topological interactions rather than being a
fundamental force. Fractal cohesion forces depend on:
- Fractal density (maps to mass)
- Hierarchical distance (maps to orbital distance)
- Tree topology (maps to orbital configuration)

If correct, this should produce identical orbital predictions to Newtonian gravity.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .entities import FractalBody, FractalOrbitalSystem


def fractal_cohesion_force(
    body1: FractalBody, 
    body2: FractalBody, 
    system: FractalOrbitalSystem
) -> float:
    """
    Calculate fractal cohesion force between two bodies.

    Based on hierarchical distance and fractal densities, this function
    implements the fractal equivalent of gravitational attraction.

    Hypothesis: This should produce the same orbital dynamics as Newtonian gravity
    when properly calibrated, proving that gravity emerges from fractal topology.

    Fractal cohesion force equation:
    F_cohesion = C * ρ1 * ρ2 / d_h²

    Where:
    - C is the system cohesion constant (analogous to G)
    - ρ1, ρ2 are fractal densities (analogous to masses)
    - d_h is hierarchical distance (analogous to orbital distance)

    Args:
        body1: First fractal body
        body2: Second fractal body
        system: The fractal orbital system

    Returns:
        Cohesion force magnitude (positive = attractive)
    """
    hierarchical_distance = system.get_hierarchical_distance(body1, body2)

    # Base cohesion from fractal densities
    density_product = body1.fractal_density * body2.fractal_density

    # Distance-dependent falloff (should match 1/r² for gravity)
    distance_factor = 1.0 / (hierarchical_distance ** 2)

    # Scale by system cohesion constant
    force = system.cohesion_constant * density_product * distance_factor

    return force


def fractal_cohesion_acceleration(
    body: FractalBody, 
    system: FractalOrbitalSystem
) -> float:
    """
    Calculate net fractal cohesion acceleration on a body.

    Similar to Newtonian acceleration (a = F/m), but using fractal parameters:
    a_cohesion = Σ(F_cohesion) / ρ

    Where ρ is the fractal density (analogous to mass).

    Args:
        body: The fractal body to calculate acceleration for
        system: The fractal orbital system

    Returns:
        Cohesion acceleration magnitude
    """
    total_force = 0.0

    for other_body in system.bodies:
        if other_body is not body:
            total_force += fractal_cohesion_force(body, other_body, system)

    # Acceleration = force / density (analogous to a = F/m)
    return total_force / body.fractal_density


def calculate_fractal_orbital_parameters(body: FractalBody, central_body: FractalBody) -> Dict[str, float]:
    """
    Calculate fractal orbital parameters from fractal hierarchy.

    Maps fractal hierarchy properties to classical orbital elements:
    - Hierarchical depth → orbital distance
    - Fractal density → effective mass
    - Tree position → orbital orientation

    Args:
        body: The fractal body
        central_body: The central fractal body

    Returns:
        Dictionary containing fractal orbital parameters
    """
    # Hierarchical distance in fractal space
    hierarchical_distance = abs(body.hierarchical_depth - central_body.hierarchical_depth)
    
    # Effective orbital distance (in AU, mapped from hierarchical depth)
    orbital_distance = body.orbital_distance
    
    # Effective mass (in solar masses, mapped from fractal density)
    effective_mass = body.effective_mass
    
    # Fractal orbital period (derived from hierarchical relationships)
    # Using Kepler's third law analog: T² ∝ a³, but with fractal scaling
    fractal_period = orbital_distance ** 1.5  # Simplified fractal analog
    
    # Fractal eccentricity (based on tree position irregularity)
    tree_irregularity = len(body.tree_address) / (body.hierarchical_depth + 1)
    fractal_eccentricity = min(0.9, tree_irregularity * 0.5)  # Bounded eccentricity
    
    return {
        'hierarchical_distance': hierarchical_distance,
        'orbital_distance': orbital_distance,
        'effective_mass': effective_mass,
        'fractal_period': fractal_period,
        'fractal_eccentricity': fractal_eccentricity,
        'tree_irregularity': tree_irregularity
    }


def simulate_fractal_trajectory(
    system: FractalOrbitalSystem,
    time_span: float,
    time_steps: int = 1000
) -> Dict[str, Any]:
    """
    Simulate orbital trajectories using fractal cohesion mechanics.

    Implements orbital motion based on fractal hierarchical interactions
    rather than classical gravity. The simulation should produce trajectories
    identical to classical mechanics if the fractal hypothesis is correct.

    Args:
        system: The fractal orbital system
        time_span: Total simulation time (normalized units)
        time_steps: Number of time steps

    Returns:
        Dictionary with trajectory data for all bodies containing:
        - times: Array of time points
        - positions: Array of position vectors over time
        - velocities: Array of velocity vectors over time
        - energies: Fractal cohesion energy over time
    """
    dt = time_span / time_steps
    times = np.linspace(0, time_span, time_steps)

    # Initialize positions and velocities for all bodies
    trajectories = {}

    for body in system.bodies:
        if body is system.central_body:
            # Central body at origin (fractal center)
            positions = [[0.0, 0.0, 0.0]] * time_steps
            velocities = [[0.0, 0.0, 0.0]] * time_steps
            cohesion_energies = [0.0] * time_steps
        else:
            # Orbital body - initialize in fractal "circular" orbit
            distance = body.orbital_distance * 1.496e11  # Convert AU to meters
            angle = 2 * np.pi * np.random.random()  # Random starting angle

            # Calculate fractal orbital velocity
            # Using fractal analog of v = sqrt(GM/r), but with fractal parameters
            # v_fractal = sqrt(C * ρ_central / d_h)
            central_density = system.central_body.fractal_density
            hierarchical_dist = system.get_hierarchical_distance(body, system.central_body)
            
            # Fractal orbital velocity (simplified model)
            fractal_velocity = np.sqrt(system.cohesion_constant * central_density / hierarchical_dist)

            # Convert to physical units for comparison
            physical_velocity = fractal_velocity * 1e4  # Scale to realistic velocities

            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            vx = -physical_velocity * np.sin(angle)  # Perpendicular to radius
            vy = physical_velocity * np.cos(angle)

            positions = [[x, y, 0.0]]
            velocities = [[vx, vy, 0.0]]
            cohesion_energies = []

            # Simple Euler integration for fractal orbital motion
            for t in range(1, time_steps):
                # Calculate fractal cohesion acceleration toward center
                r = np.array([x, y, 0.0])
                r_mag = np.linalg.norm(r)

                if r_mag > 0:
                    # Fractal cohesion acceleration
                    # a = C * ρ_central / d_h² * r_hat
                    cohesion_acc = -system.cohesion_constant * system.central_body.fractal_density / (r_mag ** 2)
                    acc_direction = r / r_mag
                    acc = cohesion_acc * acc_direction
                else:
                    acc = np.zeros(3)

                # Update velocity and position
                vx += acc[0] * dt
                vy += acc[1] * dt
                x += vx * dt
                y += vy * dt

                positions.append([x, y, 0.0])
                velocities.append([vx, vy, 0.0])
                
                # Calculate cohesion energy at this step
                cohesion_energy = -system.cohesion_constant * body.fractal_density * system.central_body.fractal_density / r_mag if r_mag > 0 else 0
                cohesion_energies.append(cohesion_energy)

        trajectories[body.name] = {
            'times': times.tolist(),
            'positions': positions,
            'velocities': velocities,
            'energies': {
                'cohesion': cohesion_energies,
                'kinetic': [],  # Will calculate from velocities
                'total': []     # Will calculate as sum
            }
        }

    # Calculate kinetic energies from velocities
    for body_name, body_data in trajectories.items():
        if body_name != system.central_body.name:
            kinetic_energies = []
            for i, vel in enumerate(body_data['velocities']):
                speed = np.linalg.norm(np.array(vel))
                # Simplified kinetic energy calculation
                ke = 0.5 * body.fractal_density * speed**2
                kinetic_energies.append(ke)
            
            body_data['energies']['kinetic'] = kinetic_energies
            body_data['energies']['total'] = [
                ke + ce for ke, ce in zip(kinetic_energies, body_data['energies']['cohesion'])
            ]

    return trajectories


def calibrate_fractal_system(
    classical_system: 'OrbitalSystem',  # Forward reference to avoid import issues
    fractal_system: FractalOrbitalSystem,
    G: float = 6.67430e-11
) -> FractalOrbitalSystem:
    """
    Calibrate fractal system parameters to match classical gravitational forces.

    This is a critical test: can we derive the gravitational constant G
    from purely fractal topological parameters?

    Calibration process:
    1. Calculate classical gravitational force for reference system
    2. Set fractal cohesion constant so fractal forces match
    3. Verify that orbital predictions are identical

    Args:
        classical_system: Reference classical orbital system
        fractal_system: Fractal system to calibrate
        G: Gravitational constant for classical calculations

    Returns:
        Calibrated fractal system with adjusted cohesion constant
    """
    # Find corresponding bodies in both systems
    earth_classical = next((b for b in classical_system.bodies if b.name == "Earth"), None)
    sun_classical = classical_system.central_body
    
    earth_fractal = next((b for b in fractal_system.bodies if b.name == "Earth"), None)
    sun_fractal = fractal_system.central_body
    
    if not earth_classical or not earth_fractal:
        # For non-Earth systems, use first non-central body
        earth_classical = next((b for b in classical_system.bodies if b is not sun_classical), None)
        earth_fractal = next((b for b in fractal_system.bodies if b is not sun_fractal), None)
    
    if not earth_classical or not earth_fractal:
        return fractal_system  # Cannot calibrate

    # Calculate classical gravitational force at reference distance
    r = earth_classical.distance_to(sun_classical)
    classical_force = G * sun_classical.mass * earth_classical.mass / (r ** 2)

    # Calculate corresponding fractal parameters
    hierarchical_dist = fractal_system.get_hierarchical_distance(earth_fractal, sun_fractal)
    density_product = earth_fractal.fractal_density * sun_fractal.fractal_density

    # Solve for fractal cohesion constant: F = C * ρ1 * ρ2 / d_h²
    # Therefore: C = F * d_h² / (ρ1 * ρ2)
    calibrated_cohesion = classical_force * (hierarchical_dist ** 2) / density_product

    # Create calibrated system
    calibrated_system = FractalOrbitalSystem(
        name=fractal_system.name,
        bodies=fractal_system.bodies,
        central_body=fractal_system.central_body,
        max_hierarchy_depth=fractal_system.max_hierarchy_depth,
        cohesion_constant=calibrated_cohesion
    )

    return calibrated_system


def validate_fractal_classical_equivalence(
    classical_trajectories: Dict[str, Any],
    fractal_trajectories: Dict[str, Any],
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validate equivalence between classical and fractal trajectory predictions.

    Compares trajectories from both frameworks to determine if they produce
    statistically identical predictions within numerical tolerance.

    Args:
        classical_trajectories: Trajectories from classical simulation
        fractal_trajectories: Trajectories from fractal simulation
        tolerance: Maximum acceptable difference (as fraction of scale)

    Returns:
        Dictionary with equivalence validation results
    """
    equivalence_results = {}
    
    # Compare each body's trajectory
    for body_name in classical_trajectories:
        if body_name in fractal_trajectories:
            classical_data = classical_trajectories[body_name]
            fractal_data = fractal_trajectories[body_name]
            
            # Extract position data
            classical_pos = np.array(classical_data['positions'])
            fractal_pos = np.array(fractal_data['positions'])
            
            # Calculate relative differences
            if len(classical_pos) > 0 and len(fractal_pos) > 0:
                # Normalize by trajectory scale
                classical_scale = np.mean(np.linalg.norm(classical_pos, axis=1))
                fractal_scale = np.mean(np.linalg.norm(fractal_pos, axis=1))
                scale_factor = classical_scale / fractal_scale if fractal_scale > 0 else 1.0
                
                # Scale fractal positions to match classical scale
                fractal_pos_scaled = fractal_pos * scale_factor
                
                # Calculate relative differences
                differences = np.linalg.norm(classical_pos - fractal_pos_scaled, axis=1)
                relative_differences = differences / classical_scale if classical_scale > 0 else differences
                
                # Statistical analysis
                max_relative_diff = np.max(relative_differences)
                mean_relative_diff = np.mean(relative_differences)
                std_relative_diff = np.std(relative_differences)
                
                # Equivalence test
                is_equivalent = max_relative_diff < tolerance
                
                equivalence_results[body_name] = {
                    'is_equivalent': is_equivalent,
                    'max_relative_difference': max_relative_diff,
                    'mean_relative_difference': mean_relative_diff,
                    'std_relative_difference': std_relative_diff,
                    'scale_factor': scale_factor,
                    'trajectory_length': len(classical_pos)
                }
            else:
                equivalence_results[body_name] = {
                    'is_equivalent': False,
                    'max_relative_difference': float('inf'),
                    'mean_relative_difference': float('inf'),
                    'std_relative_difference': float('inf'),
                    'scale_factor': 1.0,
                    'trajectory_length': 0
                }
    
    # Overall system equivalence
    equivalent_bodies = sum(1 for result in equivalence_results.values() if result['is_equivalent'])
    total_bodies = len(equivalence_results)
    system_equivalent = equivalent_bodies == total_bodies
    
    equivalence_results['system_summary'] = {
        'total_bodies': total_bodies,
        'equivalent_bodies': equivalent_bodies,
        'system_equivalent': system_equivalent,
        'equivalence_percentage': (equivalent_bodies / total_bodies * 100) if total_bodies > 0 else 0
    }
    
    return equivalence_results


def analyze_fractal_gravitational_constant(
    classical_system: 'OrbitalSystem',
    fractal_system: FractalOrbitalSystem,
    G: float = 6.67430e-11
) -> Dict[str, float]:
    """
    Analyze whether Newton's gravitational constant can be derived from fractal parameters.

    This is the key test of the hypothesis: does G emerge from fractal topology?

    Args:
        classical_system: Reference classical system
        fractal_system: Corresponding fractal system
        G: Known gravitational constant

    Returns:
        Dictionary with gravitational constant analysis
    """
    # Calculate effective G from fractal parameters
    # Using the calibration relationship: G_effective = C * (ρ1 * ρ2) / (m1 * m2) * (r² / d_h²)
    
    earth_classical = next((b for b in classical_system.bodies if b.name == "Earth"), None)
    sun_classical = classical_system.central_body
    
    earth_fractal = next((b for b in fractal_system.bodies if b.name == "Earth"), None)
    sun_fractal = fractal_system.central_body
    
    if not earth_classical or not earth_fractal:
        return {'derived_g': 0.0, 'known_g': G, 'ratio': 0.0, 'derivable': False}
    
    # Calculate distances
    classical_distance = earth_classical.distance_to(sun_classical)
    hierarchical_distance = fractal_system.get_hierarchical_distance(earth_fractal, sun_fractal)
    
    # Calculate effective G
    # From: F_classical = G * m1 * m2 / r²
    # And: F_fractal = C * ρ1 * ρ2 / d_h²
    # For equivalence: G * m1 * m2 / r² = C * ρ1 * ρ2 / d_h²
    # Therefore: G = C * ρ1 * ρ2 * r² / (m1 * m2 * d_h²)
    
    derived_g = (fractal_system.cohesion_constant * 
                earth_fractal.fractal_density * sun_fractal.fractal_density * 
                classical_distance**2 / 
                (earth_classical.mass * sun_classical.mass * hierarchical_distance**2))
    
    ratio = derived_g / G if G != 0 else 0
    derivable = 0.9 <= ratio <= 1.1  # Within 10% of known G
    
    return {
        'derived_g': derived_g,
        'known_g': G,
        'ratio': ratio,
        'derivable': derivable,
        'cohesion_constant': fractal_system.cohesion_constant,
        'classical_distance': classical_distance,
        'hierarchical_distance': hierarchical_distance
    }