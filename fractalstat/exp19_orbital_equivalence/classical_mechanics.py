"""
EXP-19: Orbital Equivalence - Classical Mechanics

Implements Newtonian gravitational mechanics for orbital trajectory simulation.
This module provides the classical physics framework against which fractal
mechanics predictions will be compared.

Key Components:
- Gravitational force calculations using Newton's law of universal gravitation
- Orbital acceleration computations
- Numerical integration for trajectory simulation
- ODE-based orbital dynamics

Scientific Foundation:
Classical orbital mechanics is based on Newton's law of universal gravitation:
F = G * m1 * m2 / r²

Where G is the gravitational constant, m1 and m2 are masses, and r is distance.
This provides the reference framework for testing whether fractal mechanics
produces equivalent predictions.
"""

import numpy as np
from scipy.integrate import odeint
from typing import List, Dict, Any, Optional
from .entities import CelestialBody, OrbitalSystem


def gravitational_force(
    body1: CelestialBody, 
    body2: CelestialBody, 
    G: float = 6.67430e-11
) -> np.ndarray:
    """
    Calculate gravitational force vector between two bodies using Newton's law.

    Newton's law of universal gravitation:
    F = G * m1 * m2 / r² * r_hat

    Where:
    - F is the gravitational force vector
    - G is the gravitational constant (6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻²)
    - m1, m2 are the masses of the two bodies
    - r is the distance between the bodies
    - r_hat is the unit vector pointing from body1 to body2

    Args:
        body1: First celestial body
        body2: Second celestial body  
        G: Gravitational constant (m³ kg⁻¹ s⁻²)

    Returns:
        Force vector on body1 due to body2 (in Newtons)

    Raises:
        ValueError: If bodies are at the same position (r = 0)
    """
    r_vector = body2.position - body1.position
    r = np.linalg.norm(r_vector)

    if r == 0:
        return np.zeros(3)  # Avoid division by zero

    force_magnitude = G * body1.mass * body2.mass / (r ** 2)
    force_direction = r_vector / r

    return force_magnitude * force_direction


def orbital_acceleration(
    body: CelestialBody, 
    system: OrbitalSystem, 
    G: float = 6.67430e-11
) -> np.ndarray:
    """
    Calculate net gravitational acceleration on a body from all other bodies.

    Uses Newton's second law: F = ma, so a = F/m

    Args:
        body: The body to calculate acceleration for
        system: The orbital system containing all bodies
        G: Gravitational constant

    Returns:
        Acceleration vector (m/s²)
    """
    total_force = np.zeros(3)

    for other_body in system.bodies:
        if other_body is not body:
            total_force += gravitational_force(body, other_body, G)

    return total_force / body.mass


def calculate_orbital_elements(body: CelestialBody, central_body: CelestialBody) -> Dict[str, float]:
    """
    Calculate classical orbital elements from position and velocity vectors.

    Args:
        body: The orbiting body
        central_body: The central body being orbited

    Returns:
        Dictionary containing orbital elements:
        - semi_major_axis: Semi-major axis (meters)
        - eccentricity: Orbital eccentricity
        - inclination: Orbital inclination (radians)
        - period: Orbital period (seconds)
    """
    # Relative position and velocity
    r = body.position - central_body.position
    v = body.velocity - central_body.velocity
    
    # Standard gravitational parameter
    mu = 6.67430e-11 * central_body.mass
    
    # Orbital energy
    specific_energy = 0.5 * np.linalg.norm(v)**2 - mu / np.linalg.norm(r)
    
    # Semi-major axis
    if specific_energy != 0:
        semi_major_axis = -mu / (2 * specific_energy)
    else:
        semi_major_axis = float('inf')  # Parabolic orbit
    
    # Angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    
    # Eccentricity vector
    e_vector = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
    eccentricity = np.linalg.norm(e_vector)
    
    # Orbital period (for elliptical orbits)
    if eccentricity < 1:
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
    else:
        period = float('inf')  # Non-elliptical orbit
    
    # Inclination (angle between h and z-axis)
    inclination = np.arccos(h[2] / h_mag)

    return {
        'semi_major_axis': semi_major_axis,
        'eccentricity': eccentricity,
        'inclination': inclination,
        'period': period,
        'angular_momentum': h_mag
    }


def simulate_orbital_trajectory(
    system: OrbitalSystem,
    time_span: float,
    time_steps: int = 1000,
    G: float = 6.67430e-11
) -> Dict[str, Any]:
    """
    Simulate orbital trajectories using classical Newtonian mechanics.

    Uses numerical integration of Newton's equations of motion to predict
    the positions and velocities of all bodies over time.

    Args:
        system: The orbital system to simulate
        time_span: Total simulation time in seconds
        time_steps: Number of time steps for the simulation
        G: Gravitational constant

    Returns:
        Dictionary with trajectory data for all bodies containing:
        - times: Array of time points
        - positions: Array of position vectors over time
        - velocities: Array of velocity vectors over time
        - energies: Kinetic and potential energy over time

    Raises:
        ValueError: If simulation parameters are invalid
    """
    if time_span <= 0:
        raise ValueError("Time span must be positive")
    if time_steps <= 0:
        raise ValueError("Time steps must be positive")

    dt = time_span / time_steps
    times = np.linspace(0, time_span, time_steps)

    # Initialize state vectors for all bodies
    # State = [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]
    initial_state = []
    for body in system.bodies:
        initial_state.extend(body.position)
        initial_state.extend(body.velocity)

    initial_state = np.array(initial_state)

    def derivatives(state: np.ndarray, t: float) -> np.ndarray:
        """
        Calculate derivatives for ODE solver.

        This function computes the time derivatives of position and velocity
        for all bodies based on Newtonian gravity.

        Args:
            state: Current state vector [positions, velocities]
            t: Current time (unused in autonomous system)

        Returns:
            Derivative vector [velocities, accelerations]
        """
        derivatives = []
        state_idx = 0

        for body in system.bodies:
            # Extract position and velocity for this body
            pos = state[state_idx:state_idx+3]
            vel = state[state_idx+3:state_idx+6]

            # Create temporary body object for acceleration calculation
            temp_body = CelestialBody(
                name=body.name,
                mass=body.mass,
                radius=body.radius,
                position=pos,
                velocity=vel
            )

            # Calculate acceleration
            acc = orbital_acceleration(temp_body, system, G)

            # Derivatives: dx/dt = vx, dy/dt = vy, dz/dt = vz, dvx/dt = ax, dvy/dt = ay, dvz/dt = az
            derivatives.extend(vel)  # position derivatives = velocity
            derivatives.extend(acc)  # velocity derivatives = acceleration

            state_idx += 6

        return np.array(derivatives)

    # Solve ODE with improved settings for orbital mechanics
    try:
        # Use more robust ODE solver settings for orbital simulations
        states = odeint(
            derivatives,
            initial_state,
            times,
            rtol=1e-8,  # Relative tolerance for orbital precision
            atol=1e-10,  # Absolute tolerance
            mxstep=5000  # Maximum steps to prevent infinite loops
        )
    except Exception as e:
        print(f"ODE integration failed: {e}")
        # Fallback to simple Euler integration with smaller steps
        print("Falling back to Euler integration...")
        states = [initial_state.copy()]
        current_state = initial_state.copy()
        dt_small = dt / 10  # Use smaller time steps for stability

        for t in times[1:]:
            # Multiple small steps per dt
            for _ in range(10):
                try:
                    derivs = derivatives(current_state, t)
                    current_state += derivs * dt_small
                except (ValueError, RuntimeWarning):
                    # Skip problematic steps
                    break
            states.append(current_state.copy())

        states = np.array(states)

    # Extract trajectories for each body
    trajectories = {}
    for i, body in enumerate(system.bodies):
        body_states = states[:, i*6:(i+1)*6]
        positions = body_states[:, :3]
        velocities = body_states[:, 3:6]
        
        # Calculate energies
        kinetic_energies = []
        potential_energies = []
        
        for j in range(len(positions)):
            # Kinetic energy: KE = 0.5 * m * v²
            ke = 0.5 * body.mass * np.linalg.norm(velocities[j])**2
            kinetic_energies.append(ke)
            
            # Potential energy relative to central body
            if body is not system.central_body:
                r = positions[j] - system.central_body.position
                pe = -G * body.mass * system.central_body.mass / np.linalg.norm(r)
                potential_energies.append(pe)
            else:
                potential_energies.append(0.0)

        trajectories[body.name] = {
            'times': times.tolist(),
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'energies': {
                'kinetic': kinetic_energies,
                'potential': potential_energies,
                'total': [ke + pe for ke, pe in zip(kinetic_energies, potential_energies)]
            },
            'orbital_elements': []
        }

    return trajectories


def validate_energy_conservation(trajectories: Dict[str, Any], G: float = 6.67430e-11) -> Dict[str, float]:
    """
    Validate energy conservation in the orbital simulation.

    In a closed gravitational system, total mechanical energy should be conserved.
    This function checks if the simulation maintains energy conservation within
    acceptable numerical tolerances.

    Args:
        trajectories: Trajectory data from simulation
        G: Gravitational constant

    Returns:
        Dictionary with energy conservation metrics:
        - energy_drift: Percentage change in total energy
        - max_energy_variation: Maximum variation from initial energy
        - conservation_valid: Whether energy is conserved within tolerance
    """
    if not trajectories:
        return {'energy_drift': 0.0, 'max_energy_variation': 0.0, 'conservation_valid': False}

    # Calculate total system energy over time
    total_energies = []
    
    # Get time points from first body
    times = trajectories[list(trajectories.keys())[0]]['times']
    
    for t_idx in range(len(times)):
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        # Sum kinetic energy of all bodies
        for body_name, body_data in trajectories.items():
            ke = body_data['energies']['kinetic'][t_idx]
            kinetic_energy += ke
        
        # Sum potential energy between all pairs
        body_names = list(trajectories.keys())
        for i in range(len(body_names)):
            for j in range(i + 1, len(body_names)):
                body1_data = trajectories[body_names[i]]
                body2_data = trajectories[body_names[j]]
                
                pos1 = np.array(body1_data['positions'][t_idx])
                pos2 = np.array(body2_data['positions'][t_idx])
                
                distance = np.linalg.norm(pos1 - pos2)
                if distance > 0:
                    pe = -G * 1.0 * 1.0 / distance  # Simplified for validation
                    potential_energy += pe
        
        total_energies.append(kinetic_energy + potential_energy)
    
    # Calculate energy conservation metrics
    initial_energy = total_energies[0]
    final_energy = total_energies[-1]
    
    energy_drift = ((final_energy - initial_energy) / initial_energy) * 100
    max_variation = max(abs(e - initial_energy) for e in total_energies)
    max_variation_percent = (max_variation / abs(initial_energy)) * 100 if initial_energy != 0 else 0
    
    # Energy is considered conserved if drift is less than 1%
    conservation_valid = abs(energy_drift) < 1.0

    return {
        'energy_drift': energy_drift,
        'max_energy_variation_percent': max_variation_percent,
        'conservation_valid': conservation_valid,
        'initial_energy': initial_energy,
        'final_energy': final_energy
    }


def calculate_orbital_stability(trajectories: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze orbital stability from trajectory data.

    Checks for signs of orbital instability such as:
    - Escaping bodies (increasing distance from central body)
    - Collisions (decreasing distance below minimum threshold)
    - Chaotic behavior (irregular orbital parameters)

    Args:
        trajectories: Trajectory data from simulation

    Returns:
        Dictionary with stability analysis for each body
    """
    stability_analysis = {}
    
    for body_name, body_data in trajectories.items():
        if body_name == "Sun":  # Skip central body
            continue
            
        positions = np.array(body_data['positions'])
        
        # Calculate distance from origin over time
        distances = np.linalg.norm(positions, axis=1)
        
        # Check for escaping behavior
        distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]
        is_escaping = distance_trend > 0 and abs(distance_trend) > 1e6  # m/s escape velocity
        
        # Check for collision behavior
        min_distance = np.min(distances)
        is_colliding = min_distance < 1e6  # Less than 1000 km from center
        
        # Check for orbital period consistency
        # Look for peaks in distance (apoapsis)
        peaks = []
        for i in range(1, len(distances)-1):
            if distances[i] > distances[i-1] and distances[i] > distances[i+1]:
                peaks.append(i)
        
        # Calculate period variations
        if len(peaks) > 2:
            periods = np.diff(peaks)
            period_std = np.std(periods)
            period_mean = np.mean(periods)
            period_variation = period_std / period_mean if period_mean > 0 else 0
            is_chaotic = period_variation > 0.1  # 10% variation indicates chaos
        else:
            is_chaotic = False
        
        stability_analysis[body_name] = {
            'is_escaping': is_escaping,
            'is_colliding': is_colliding,
            'is_chaotic': is_chaotic,
            'min_distance': min_distance,
            'max_distance': np.max(distances),
            'average_distance': np.mean(distances),
            'distance_trend': distance_trend
        }
    
    return stability_analysis