"""
EXP-19: Orbital Equivalence - Trajectory Comparison

Implements comprehensive comparison between classical and fractal trajectory
predictions to validate the orbital equivalence hypothesis.

Key Components:
- TrajectoryComparison: Detailed comparison between frameworks
- OrbitalEquivalenceTest: Overall equivalence testing
- Statistical analysis of prediction differences
- Orbital period and correlation analysis

Scientific Foundation:
The comparison module validates whether two fundamentally different
mathematical frameworks (Newtonian gravity vs fractal cohesion) produce
statistically identical predictions for the same physical systems.

Success Criteria:
- Position correlation > 0.99 between frameworks
- Trajectory similarity > 0.99 
- Identical orbital period predictions
- Identical responses to perturbations
"""

import numpy as np
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .entities import FractalBody, FractalOrbitalSystem


@dataclass
class TrajectoryComparison:
    """Comparison between classical and fractal trajectory predictions.

    Provides comprehensive analysis of how well two trajectory predictions
    match across multiple metrics including position correlation, trajectory
    similarity, and orbital period consistency.

    Attributes:
        body_name: Name of the celestial body being compared
        classical_positions: Position data from classical simulation
        fractal_positions: Position data from fractal simulation
        times: Time points for the comparison
    """

    body_name: str
    classical_positions: List[List[float]]
    fractal_positions: List[List[float]]
    times: List[float]

    # Comparison metrics (calculated in post_init)
    position_correlation: float = field(init=False, default=0.0)
    trajectory_similarity: float = field(init=False, default=0.0)
    orbital_period_match: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Calculate comparison metrics after initialization."""
        self._calculate_position_correlation()
        self._calculate_trajectory_similarity()
        self._calculate_orbital_period_match()

    def _calculate_position_correlation(self):
        """Calculate correlation between predicted positions.

        Uses Pearson correlation coefficient to measure linear relationship
        between classical and fractal position predictions for each coordinate axis.
        """
        if not self.classical_positions or not self.fractal_positions:
            self.position_correlation = 0.0
            return

        # Convert to numpy arrays
        classical = np.array(self.classical_positions)
        fractal = np.array(self.fractal_positions)

        # Calculate correlation for each coordinate
        correlations = []
        for i in range(3):  # x, y, z coordinates
            if len(classical) > 1 and len(fractal) > 1:
                try:
                    # Check for valid data (not all zeros, not NaN)
                    classical_coord = classical[:, i]
                    fractal_coord = fractal[:, i]

                    if (np.any(classical_coord != 0) and np.any(fractal_coord != 0) and
                        not np.any(np.isnan(classical_coord)) and not np.any(np.isnan(fractal_coord)) and
                        np.std(classical_coord) > 0 and np.std(fractal_coord) > 0):

                        corr = np.corrcoef(classical_coord, fractal_coord)[0, 1]
                        if not np.isnan(corr) and np.isfinite(corr):
                            correlations.append(abs(corr))
                except (ValueError, TypeError, IndexError, RuntimeWarning):
                    # Silently skip problematic coordinates
                    pass

        self.position_correlation = statistics.mean(correlations) if correlations else 0.0

    def _calculate_trajectory_similarity(self):
        """Calculate overall trajectory similarity using Euclidean distance.

        Measures how closely the two predicted trajectories match in 3D space
        by calculating the average Euclidean distance between corresponding points.
        """
        if not self.classical_positions or not self.fractal_positions:
            self.trajectory_similarity = 0.0
            return

        classical = np.array(self.classical_positions)
        fractal = np.array(self.fractal_positions)

        # Check for valid data
        if (np.any(np.isnan(classical)) or np.any(np.isnan(fractal)) or
            np.any(np.isinf(classical)) or np.any(np.isinf(fractal))):
            self.trajectory_similarity = 0.0
            return

        # Calculate average Euclidean distance between trajectories
        distances = []
        min_len = min(len(classical), len(fractal))

        for i in range(min_len):
            try:
                dist = np.linalg.norm(classical[i] - fractal[i])
                if np.isfinite(dist):
                    distances.append(dist)
            except (ValueError, RuntimeWarning):
                continue

        if distances:
            # Similarity = 1 / (1 + average_distance)
            # Normalize by trajectory scale
            avg_distance = statistics.mean(distances)
            trajectory_scale = np.mean([np.linalg.norm(pos) for pos in classical[:min_len] if np.all(np.isfinite(pos))])

            if trajectory_scale > 0 and np.isfinite(avg_distance):
                normalized_distance = avg_distance / trajectory_scale
                if np.isfinite(normalized_distance):
                    self.trajectory_similarity = 1.0 / (1.0 + normalized_distance)
                else:
                    self.trajectory_similarity = 0.0
            else:
                self.trajectory_similarity = 0.0
        else:
            self.trajectory_similarity = 0.0

    def _calculate_orbital_period_match(self):
        """Calculate how well orbital periods match between frameworks.

        Analyzes the periodic nature of the trajectories by examining
        radial distance oscillations to determine orbital periods.
        """
        # Simplified: check if radial distance oscillations match
        if not self.classical_positions or not self.fractal_positions:
            self.orbital_period_match = 0.0
            return

        def calculate_periodic_signature(positions):
            """Calculate periodic signature from radial distance.

            Identifies peaks in the radial distance (apoapsis) to determine
            the periodicity of the orbit.
            """
            radii = [np.linalg.norm(pos) for pos in positions]
            # Find peaks in radius (apoapsis/periapsis)
            peaks = []
            for i in range(1, len(radii)-1):
                if radii[i] > radii[i-1] and radii[i] > radii[i+1]:
                    peaks.append(i)
            return len(peaks) / len(radii) if radii else 0

        classical_signature = calculate_periodic_signature(self.classical_positions)
        fractal_signature = calculate_periodic_signature(self.fractal_positions)

        # Similarity of periodic signatures
        self.orbital_period_match = 1.0 - abs(classical_signature - fractal_signature)

    def get_comparison_summary(self) -> Dict[str, float]:
        """Get summary of all comparison metrics."""
        return {
            'body_name': self.body_name,
            'position_correlation': self.position_correlation,
            'trajectory_similarity': self.trajectory_similarity,
            'orbital_period_match': self.orbital_period_match,
            'overall_score': (self.position_correlation + self.trajectory_similarity + self.orbital_period_match) / 3
        }

    def is_equivalent(self, correlation_threshold: float = 0.99) -> bool:
        """Determine if trajectories are equivalent within threshold.

        Args:
            correlation_threshold: Minimum correlation for equivalence

        Returns:
            True if trajectories are equivalent
        """
        return (self.position_correlation >= correlation_threshold and
                self.trajectory_similarity >= correlation_threshold and
                self.orbital_period_match >= correlation_threshold)


@dataclass
class OrbitalEquivalenceTest:
    """Results from testing orbital equivalence between frameworks.

    Provides comprehensive analysis of whether classical and fractal mechanics
    produce equivalent predictions across multiple celestial bodies and metrics.

    Attributes:
        system_name: Name of the orbital system being tested
        classical_trajectories: Trajectory data from classical simulation
        fractal_trajectories: Trajectory data from fractal simulation
        comparisons: Individual body comparisons
    """

    system_name: str
    classical_trajectories: Dict[str, Any]
    fractal_trajectories: Dict[str, Any]
    comparisons: Dict[str, TrajectoryComparison]

    # Overall equivalence metrics (calculated in post_init)
    average_position_correlation: float = field(init=False, default=0.0)
    average_trajectory_similarity: float = field(init=False, default=0.0)
    average_orbital_period_match: float = field(init=False, default=0.0)
    equivalence_confirmed: bool = field(init=False, default=False)

    def __post_init__(self):
        """Calculate overall equivalence metrics after initialization."""
        self._calculate_overall_metrics()

    def _calculate_overall_metrics(self):
        """Calculate aggregate metrics across all bodies."""
        if not self.comparisons:
            self.average_position_correlation = 0.0
            self.average_trajectory_similarity = 0.0
            self.average_orbital_period_match = 0.0
            self.equivalence_confirmed = False
            return

        correlations = [comp.position_correlation for comp in self.comparisons.values()]
        similarities = [comp.trajectory_similarity for comp in self.comparisons.values()]
        period_matches = [comp.orbital_period_match for comp in self.comparisons.values()]

        self.average_position_correlation = statistics.mean(correlations)
        self.average_trajectory_similarity = statistics.mean(similarities)
        self.average_orbital_period_match = statistics.mean(period_matches)

        # Equivalence confirmed if all metrics > 0.99
        self.equivalence_confirmed = (
            self.average_position_correlation > 0.99 and
            self.average_trajectory_similarity > 0.99 and
            self.average_orbital_period_match > 0.99
        )

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of system equivalence test."""
        equivalent_bodies = sum(1 for comp in self.comparisons.values() if comp.is_equivalent())
        total_bodies = len(self.comparisons)

        return {
            'system_name': self.system_name,
            'total_bodies': total_bodies,
            'equivalent_bodies': equivalent_bodies,
            'equivalence_percentage': (equivalent_bodies / total_bodies * 100) if total_bodies > 0 else 0,
            'average_position_correlation': self.average_position_correlation,
            'average_trajectory_similarity': self.average_trajectory_similarity,
            'average_orbital_period_match': self.average_orbital_period_match,
            'equivalence_confirmed': self.equivalence_confirmed,
            'body_results': {
                name: comp.get_comparison_summary() for name, comp in self.comparisons.items()
            }
        }

    def get_equivalence_report(self) -> str:
        """Generate human-readable equivalence report."""
        summary = self.get_system_summary()

        report = f"""
ORBITAL EQUIVALENCE TEST REPORT: {self.system_name}
{'=' * 60}

SYSTEM OVERVIEW:
- Total bodies tested: {summary['total_bodies']}
- Equivalent bodies: {summary['equivalent_bodies']}
- Equivalence percentage: {summary['equivalence_percentage']:.1f}%

EQUIVALENCE METRICS:
- Average position correlation: {summary['average_position_correlation']:.6f}
- Average trajectory similarity: {summary['average_trajectory_similarity']:.6f}
- Average orbital period match: {summary['average_orbital_period_match']:.6f}

OVERALL RESULT: {'PASSED' if summary['equivalence_confirmed'] else 'FAILED'}

BODY-BY-BODY RESULTS:
"""

        for body_name, body_result in summary['body_results'].items():
            status = "✓ EQUIVALENT" if body_result['overall_score'] > 0.99 else "✗ NOT EQUIVALENT"
            report += f"""
{body_name}: {status}
  - Position correlation: {body_result['position_correlation']:.6f}
  - Trajectory similarity: {body_result['trajectory_similarity']:.6f}
  - Period match: {body_result['orbital_period_match']:.6f}
  - Overall score: {body_result['overall_score']:.6f}
"""

        if summary['equivalence_confirmed']:
            report += """
BREAKTHROUGH CONFIRMED:
Orbital mechanics IS fractal mechanics under a different representation!
Newtonian gravity emerges from fractal hierarchical topology.
"""
        else:
            report += """
EQUIVALENCE NOT CONFIRMED:
Further investigation needed to understand discrepancies between frameworks.
"""

        return report

    def analyze_perturbation_response(self, perturbed_comparisons: Dict[str, TrajectoryComparison]) -> Dict[str, Any]:
        """Analyze how both frameworks respond to perturbations.

        Tests whether classical and fractal mechanics respond identically
        to external disturbances (e.g., rogue planet encounters).

        Args:
            perturbed_comparisons: Comparisons after perturbation

        Returns:
            Analysis of perturbation response equivalence
        """
        if not perturbed_comparisons:
            return {'perturbation_equivalence': False, 'analysis': 'No perturbation data available'}

        # Compare pre- and post-perturbation for each body
        perturbation_analysis = {}

        for body_name in self.comparisons:
            if body_name in perturbed_comparisons:
                original_comp = self.comparisons[body_name]
                perturbed_comp = perturbed_comparisons[body_name]

                # Calculate change in metrics due to perturbation
                correlation_change = perturbed_comp.position_correlation - original_comp.position_correlation
                similarity_change = perturbed_comp.trajectory_similarity - original_comp.trajectory_similarity
                period_change = perturbed_comp.orbital_period_match - original_comp.orbital_period_match

                perturbation_analysis[body_name] = {
                    'correlation_change': correlation_change,
                    'similarity_change': similarity_change,
                    'period_change': period_change,
                    'response_consistent': (
                        abs(correlation_change) < 0.01 and
                        abs(similarity_change) < 0.01 and
                        abs(period_change) < 0.01
                    )
                }

        # Overall perturbation equivalence
        consistent_responses = sum(1 for analysis in perturbation_analysis.values() if analysis['response_consistent'])
        total_responses = len(perturbation_analysis)
        perturbation_equivalence = consistent_responses == total_responses

        return {
            'perturbation_equivalence': perturbation_equivalence,
            'consistent_responses': consistent_responses,
            'total_responses': total_responses,
            'response_analysis': perturbation_analysis
        }

    def validate_energy_conservation(self, classical_energies: Dict[str, List[float]], fractal_energies: Dict[str, List[float]]) -> Dict[str, Any]:
        """Validate energy conservation in both frameworks.

        Checks if both classical and fractal simulations conserve energy
        and whether their energy conservation patterns match.

        Args:
            classical_energies: Energy data from classical simulation
            fractal_energies: Energy data from fractal simulation

        Returns:
            Energy conservation validation results
        """
        energy_analysis = {}

        for body_name in classical_energies:
            if body_name in fractal_energies:
                classical_energy = classical_energies[body_name]
                fractal_energy = fractal_energies[body_name]

                # Calculate energy drift
                classical_drift = (classical_energy[-1] - classical_energy[0]) / classical_energy[0] if classical_energy[0] != 0 else 0
                fractal_drift = (fractal_energy[-1] - fractal_energy[0]) / fractal_energy[0] if fractal_energy[0] != 0 else 0

                # Energy conservation similarity
                energy_conservation_similar = abs(classical_drift - fractal_drift) < 0.01

                energy_analysis[body_name] = {
                    'classical_energy_drift': classical_drift,
                    'fractal_energy_drift': fractal_drift,
                    'energy_conservation_similar': energy_conservation_similar,
                    'classical_conserved': abs(classical_drift) < 0.01,
                    'fractal_conserved': abs(fractal_drift) < 0.01
                }

        # Overall energy conservation equivalence
        similar_conservation = sum(1 for analysis in energy_analysis.values() if analysis['energy_conservation_similar'])
        total_bodies = len(energy_analysis)
        energy_equivalence = similar_conservation == total_bodies

        return {
            'energy_equivalence': energy_equivalence,
            'similar_conservation': similar_conservation,
            'total_bodies': total_bodies,
            'energy_analysis': energy_analysis
        }