"""
Trajectory computation and comparison for vector field validation.

Contains classes and functions for computing orbital trajectories using vector
fields and comparing them with Newtonian mechanics for validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .entities import FractalEntity
from .vector_field_system import VectorFieldApproach


@dataclass
class OrbitalTrajectory:
    """Trajectory computed using vector field."""

    approach_name: str
    times: np.ndarray
    positions: List[np.ndarray]  # List of [x,y,z] positions
    velocities: List[np.ndarray]  # List of [vx,vy,vz] velocities
    energies: List[float]  # Total energy at each time step

    # Trajectory quality metrics
    trajectory_similarity: float = field(init=False)
    period_accuracy: float = field(init=False)
    position_correlation: float = field(init=False)

    def __post_init__(self):
        """Calculate trajectory quality metrics."""
        # These would be calculated by comparing to Newtonian trajectory
        self.trajectory_similarity = 0.0  # Placeholder
        self.period_accuracy = 0.0       # Placeholder
        self.position_correlation = 0.0  # Placeholder


@dataclass
class TrajectoryComparison:
    """Comparison between fractal and Newtonian trajectories."""

    system_name: str
    approach_name: str
    fractal_trajectory: OrbitalTrajectory
    newtonian_trajectory: OrbitalTrajectory

    # Comparison metrics
    position_correlation: float = field(init=False)
    trajectory_similarity: float = field(init=False)
    period_accuracy: float = field(init=False)
    energy_conservation: float = field(init=False)

    def __post_init__(self):
        """Calculate comparison metrics."""
        self._calculate_position_correlation()
        self._calculate_trajectory_similarity()
        self._calculate_period_accuracy()
        self._calculate_energy_conservation()

    def _calculate_position_correlation(self):
        """Calculate correlation between predicted positions."""
        if not self.fractal_trajectory.positions or not self.newtonian_trajectory.positions:
            self.position_correlation = 0.0
            return

        fractal_pos = np.array(self.fractal_trajectory.positions)
        newtonian_pos = np.array(self.newtonian_trajectory.positions)

        correlations = []
        for i in range(3):  # x, y, z coordinates
            try:
                corr = np.corrcoef(fractal_pos[:, i], newtonian_pos[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except (ValueError, TypeError, IndexError) as e:
                print(f"Warning: Could not compute position correlation for coordinate {i}: {e}")
                pass

        self.position_correlation = np.mean(correlations) if correlations else 0.0

    def _calculate_trajectory_similarity(self):
        """Calculate overall trajectory similarity using Euclidean distance."""
        if not self.fractal_trajectory.positions or not self.newtonian_trajectory.positions:
            self.trajectory_similarity = 0.0
            return

        fractal_pos = np.array(self.fractal_trajectory.positions)
        newtonian_pos = np.array(self.newtonian_trajectory.positions)

        distances = []
        min_len = min(len(fractal_pos), len(newtonian_pos))

        for i in range(min_len):
            dist = np.linalg.norm(fractal_pos[i] - newtonian_pos[i])
            distances.append(dist)

        if distances:
            avg_distance = np.mean(distances)
            # Normalize by trajectory scale
            trajectory_scale = np.mean([np.linalg.norm(pos) for pos in newtonian_pos[:min_len]])
            if trajectory_scale > 0:
                normalized_distance = avg_distance / trajectory_scale
                self.trajectory_similarity = 1.0 / (1.0 + normalized_distance)
            else:
                self.trajectory_similarity = 0.0
        else:
            self.trajectory_similarity = 0.0

    def _calculate_period_accuracy(self):
        """Calculate how well orbital periods match."""
        # Simplified: compare radial oscillation patterns
        def extract_periodic_signal(positions):
            radii = [np.linalg.norm(pos) for pos in positions]
            # Count peaks in radius (apoapsis/periapsis)
            peaks = 0
            for i in range(1, len(radii)-1):
                if radii[i] > radii[i-1] and radii[i] > radii[i+1]:
                    peaks += 1
            return peaks / len(radii) if radii else 0

        fractal_signal = extract_periodic_signal(self.fractal_trajectory.positions)
        newtonian_signal = extract_periodic_signal(self.newtonian_trajectory.positions)

        if newtonian_signal > 0:
            self.period_accuracy = 1.0 - abs(fractal_signal - newtonian_signal) / newtonian_signal
            self.period_accuracy = max(0.0, min(1.0, self.period_accuracy))
        else:
            self.period_accuracy = 0.0

    def _calculate_energy_conservation(self):
        """Calculate how well energy is conserved in fractal trajectory."""
        energies = self.fractal_trajectory.energies
        if not energies:
            self.energy_conservation = 0.0
            return

        initial_energy = energies[0]
        final_energy = energies[-1]
        max_energy = max(energies)
        min_energy = min(energies)

        # Energy conservation metric
        energy_range = max_energy - min_energy
        if energy_range > 0:
            self.energy_conservation = 1.0 - abs(final_energy - initial_energy) / energy_range
            self.energy_conservation = max(0.0, self.energy_conservation)
        else:
            self.energy_conservation = 1.0  # Perfectly constant energy


def integrate_orbit_with_vector_field(
    entity_a: FractalEntity,
    entity_b: FractalEntity,
    vector_approach: VectorFieldApproach,
    scalar_magnitude: float,
    time_span: float,
    time_steps: int = 1000
) -> OrbitalTrajectory:
    """
    Integrate orbital trajectory using derived vector field.

    Args:
        entity_a, entity_b: The two orbiting entities
        vector_approach: Which vector derivation approach to use
        scalar_magnitude: Base force magnitude at reference distance
        time_span: Total integration time
        time_steps: Number of time steps

    Returns:
        Computed orbital trajectory
    """
    dt = time_span / time_steps
    times = np.linspace(0, time_span, time_steps)

    # Reference distance (1 AU for Earth-Sun)
    reference_distance = 1.496e11  # meters

    # Initial conditions
    positions = [entity_a.position.copy()]
    velocities = [entity_a.velocity.copy()]
    energies = []

    current_pos = entity_a.position.copy()
    current_vel = entity_a.velocity.copy()

    for i in range(1, time_steps):
        # Calculate current distance from central body
        r_vector = entity_b.position - current_pos
        current_distance = np.linalg.norm(r_vector)

        # Calculate force magnitude with inverse-square falloff
        # F = F_ref * (r_ref / r)^2
        if current_distance > 0:
            distance_factor = (reference_distance / current_distance) ** 2
            effective_magnitude = scalar_magnitude * distance_factor
        else:
            effective_magnitude = scalar_magnitude

        # Create temporary entities for force calculation
        temp_a = FractalEntity(
            name=entity_a.name,
            position=current_pos,
            velocity=current_vel,
            mass=entity_a.mass,
            fractal_density=entity_a.fractal_density,
            hierarchical_depth=entity_a.hierarchical_depth,
            branching_factor=entity_a.branching_factor
        )

        # Derive force vector with distance-dependent magnitude
        force_vector = vector_approach.derive_force(temp_a, entity_b, effective_magnitude)

        # Calculate acceleration
        acceleration = force_vector / entity_a.mass

        # Verlet integration (more stable than Euler)
        new_pos = current_pos + current_vel * dt + 0.5 * acceleration * dt**2
        new_acc = acceleration  # Simplified - would recalculate at new position

        new_vel = current_vel + 0.5 * (acceleration + new_acc) * dt

        # Store results
        positions.append(new_pos.copy())
        velocities.append(new_vel.copy())

        # Calculate energy (kinetic + potential approximation)
        kinetic = 0.5 * entity_a.mass * np.linalg.norm(current_vel)**2
        r = current_distance
        potential = -scalar_magnitude * reference_distance**2 * entity_b.mass / (r**2) if r > 0 else 0
        energies.append(kinetic + potential)

        # Update for next iteration
        current_pos = new_pos
        current_vel = new_vel

    return OrbitalTrajectory(
        approach_name=vector_approach.name,
        times=times,
        positions=positions,
        velocities=velocities,
        energies=energies
    )


def compute_newtonian_trajectory(
    orbiting_body: FractalEntity,
    central_body: FractalEntity,
    times: np.ndarray
) -> OrbitalTrajectory:
    """
    Compute Newtonian trajectory for comparison.

    Args:
        orbiting_body, central_body: The two bodies
        times: Time points for trajectory

    Returns:
        Newtonian trajectory
    """
    G = 6.67430e-11  # Gravitational constant

    positions = []
    velocities = []
    energies = []

    # Initial conditions
    pos = orbiting_body.position.copy()
    vel = orbiting_body.velocity.copy()

    dt = times[1] - times[0] if len(times) > 1 else 1.0

    for t in times:
        positions.append(pos.copy())
        velocities.append(vel.copy())

        # Calculate gravitational acceleration
        r_vector = central_body.position - pos
        r = np.linalg.norm(r_vector)

        if r > 0:
            acceleration = G * central_body.mass * r_vector / (r ** 3)
        else:
            acceleration = np.zeros(3)

        # Euler integration
        vel += acceleration * dt
        pos += vel * dt

        # Calculate energy
        kinetic = 0.5 * orbiting_body.mass * np.linalg.norm(vel)**2
        potential = -G * central_body.mass * orbiting_body.mass / r if r > 0 else 0
        energies.append(kinetic + potential)

    return OrbitalTrajectory(
        approach_name="Newtonian",
        times=times,
        positions=positions,
        velocities=velocities,
        energies=energies
    )