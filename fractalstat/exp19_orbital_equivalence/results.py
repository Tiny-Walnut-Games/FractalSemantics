"""
EXP-19: Orbital Equivalence - Results Processing and File I/O

Handles saving, loading, and analyzing experimental results from orbital
equivalence tests. Provides comprehensive reporting and data persistence
for the experiment results.

Key Features:
- Save/load experiment results to/from JSON files
- Generate comprehensive reports
- Statistical analysis of equivalence metrics
- Visualization data preparation
- Results comparison across multiple runs

Scientific Foundation:
This module ensures that the critical experimental results from testing
the orbital equivalence hypothesis are properly preserved, analyzed,
and reported for scientific validation and peer review.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from .comparison import OrbitalEquivalenceTest, TrajectoryComparison


@dataclass
class EXP19_Results:
    """Complete results from EXP-19 orbital equivalence test.

    Contains all data and analysis from a complete orbital equivalence experiment,
    including trajectory data, comparison metrics, validation results, and metadata.

    Attributes:
        experiment_name: Name of the experiment
        timestamp: When the experiment was run
        system_name: Name of the orbital system tested
        simulation_parameters: Parameters used for the simulation
        classical_trajectories: Trajectory data from classical simulation
        fractal_trajectories: Trajectory data from fractal simulation
        comparisons: Individual body comparisons
        equivalence_test: Overall equivalence test results
        energy_analysis: Energy conservation analysis
        perturbation_analysis: Perturbation response analysis (if applicable)
        validation_results: System validation results
    """

    experiment_name: str
    timestamp: str
    system_name: str
    simulation_parameters: Dict[str, Any]
    classical_trajectories: Dict[str, Any]
    fractal_trajectories: Dict[str, Any]
    comparisons: Dict[str, TrajectoryComparison]
    equivalence_test: OrbitalEquivalenceTest
    energy_analysis: Dict[str, Any]
    perturbation_analysis: Optional[Dict[str, Any]]
    validation_results: Dict[str, Any]


def save_results(results: EXP19_Results, output_file: Optional[str] = None) -> str:
    """Save results to JSON file.

    Args:
        results: Complete experiment results
        output_file: Output file path (auto-generated if not provided)

    Returns:
        Path to saved file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"exp19_results_{results.system_name.replace(' ', '_')}_{timestamp}.json"

    # Convert dataclasses to dictionaries for JSON serialization
    results_dict = {
        'experiment_name': results.experiment_name,
        'timestamp': results.timestamp,
        'system_name': results.system_name,
        'simulation_parameters': results.simulation_parameters,
        'classical_trajectories': results.classical_trajectories,
        'fractal_trajectories': results.fractal_trajectories,
        'comparisons': {name: asdict(comp) for name, comp in results.comparisons.items()},
        'equivalence_test': {
            'system_name': results.equivalence_test.system_name,
            'average_position_correlation': results.equivalence_test.average_position_correlation,
            'average_trajectory_similarity': results.equivalence_test.average_trajectory_similarity,
            'average_orbital_period_match': results.equivalence_test.average_orbital_period_match,
            'equivalence_confirmed': results.equivalence_test.equivalence_confirmed,
            'body_results': {name: comp.get_comparison_summary() for name, comp in results.equivalence_test.comparisons.items()}
        },
        'energy_analysis': results.energy_analysis,
        'perturbation_analysis': results.perturbation_analysis,
        'validation_results': results.validation_results
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    return output_file


def load_results(input_file: str) -> EXP19_Results:
    """Load results from JSON file.

    Args:
        input_file: Path to JSON file containing results

    Returns:
        Complete experiment results
    """
    with open(input_file, 'r') as f:
        results_dict = json.load(f)

    # Reconstruct dataclasses from dictionaries
    comparisons = {}
    for name, comp_dict in results_dict['comparisons'].items():
        comparisons[name] = TrajectoryComparison(
            body_name=comp_dict['body_name'],
            classical_positions=comp_dict['classical_positions'],
            fractal_positions=comp_dict['fractal_positions'],
            times=comp_dict['times'],
            position_correlation=comp_dict['position_correlation'],
            trajectory_similarity=comp_dict['trajectory_similarity'],
            orbital_period_match=comp_dict['orbital_period_match']
        )

    equivalence_test = OrbitalEquivalenceTest(
        system_name=results_dict['equivalence_test']['system_name'],
        classical_trajectories=results_dict['classical_trajectories'],
        fractal_trajectories=results_dict['fractal_trajectories'],
        comparisons=comparisons
    )

    return EXP19_Results(
        experiment_name=results_dict['experiment_name'],
        timestamp=results_dict['timestamp'],
        system_name=results_dict['system_name'],
        simulation_parameters=results_dict['simulation_parameters'],
        classical_trajectories=results_dict['classical_trajectories'],
        fractal_trajectories=results_dict['fractal_trajectories'],
        comparisons=comparisons,
        equivalence_test=equivalence_test,
        energy_analysis=results_dict['energy_analysis'],
        perturbation_analysis=results_dict['perturbation_analysis'],
        validation_results=results_dict['validation_results']
    )


def generate_results_report(results: EXP19_Results) -> str:
    """Generate comprehensive results report.

    Args:
        results: Complete experiment results

    Returns:
        Formatted report string
    """
    report = f"""
EXP-19 ORBITAL EQUIVALENCE TEST REPORT
{'=' * 80}

EXPERIMENT INFORMATION:
- Experiment: {results.experiment_name}
- System: {results.system_name}
- Timestamp: {results.timestamp}
- Simulation Time: {results.simulation_parameters.get('simulation_time', 'N/A')} seconds
- Time Steps: {results.simulation_parameters.get('time_steps', 'N/A')}

SYSTEM VALIDATION:
- Body mapping valid: {'✓' if results.validation_results['body_mapping_valid'] else '✗'}
- Mass ratio preserved: {'✓' if results.validation_results['mass_ratio_preserved'] else '✗'}
- Distance mapping valid: {'✓' if results.validation_results['distance_mapping_valid'] else '✗'}

EQUIVALENCE TEST RESULTS:
- Equivalence confirmed: {'✓ YES' if results.equivalence_test.equivalence_confirmed else '✗ NO'}
- Average position correlation: {results.equivalence_test.average_position_correlation:.6f}
- Average trajectory similarity: {results.equivalence_test.average_trajectory_similarity:.6f}
- Average orbital period match: {results.equivalence_test.average_orbital_period_match:.6f}

BODY-BY-BODY ANALYSIS:
"""

    for body_name, body_result in results.equivalence_test.get_system_summary()['body_results'].items():
        status = "✓ EQUIVALENT" if body_result['overall_score'] > 0.99 else "✗ NOT EQUIVALENT"
        report += f"""
{body_name}: {status}
  - Position correlation: {body_result['position_correlation']:.6f}
  - Trajectory similarity: {body_result['trajectory_similarity']:.6f}
  - Period match: {body_result['orbital_period_match']:.6f}
  - Overall score: {body_result['overall_score']:.6f}
"""

    # Energy analysis
    report += f"""
ENERGY CONSERVATION ANALYSIS:
- Energy equivalence: {'✓ YES' if results.energy_analysis['energy_equivalence'] else '✗ NO'}
- Similar conservation patterns: {results.energy_analysis['similar_conservation']}/{results.energy_analysis['total_bodies']}
"""

    # Perturbation analysis
    if results.perturbation_analysis:
        report += f"""
PERTURBATION RESPONSE ANALYSIS:
- Perturbation equivalence: {'✓ YES' if results.perturbation_analysis['perturbation_equivalence'] else '✗ NO'}
- Consistent responses: {results.perturbation_analysis['consistent_responses']}/{results.perturbation_analysis['total_responses']}
"""
    else:
        report += """
PERTURBATION RESPONSE ANALYSIS:
- Not tested (perturbation analysis not included)
"""

    # Final conclusion
    if results.equivalence_test.equivalence_confirmed:
        report += """
BREAKTHROUGH CONFIRMED:
Orbital mechanics IS fractal mechanics under a different representation!
Newtonian gravity emerges from fractal hierarchical topology.

This represents a fundamental unification of classical and quantum descriptions
of gravity, with profound implications for theoretical physics.
"""
    else:
        report += """
EQUIVALENCE NOT CONFIRMED:
The hypothesis requires further investigation and potential revision.

Possible issues:
- Fractal parameter calibration may need refinement
- Simulation numerical precision may be insufficient
- Fundamental assumptions about fractal-gravity relationship may be incorrect
"""

    report += f"""
{'=' * 80}
END OF REPORT
{'=' * 80}
"""

    return report


def compare_results(results_list: List[EXP19_Results]) -> Dict[str, Any]:
    """Compare multiple experiment results.

    Args:
        results_list: List of experiment results to compare

    Returns:
        Comparison analysis across all results
    """
    if not results_list:
        return {'error': 'No results provided for comparison'}

    comparison = {
        'total_experiments': len(results_list),
        'systems_tested': [],
        'equivalence_confirmed': 0,
        'average_correlations': [],
        'average_similarities': [],
        'average_period_matches': [],
        'success_rates': {}
    }

    for results in results_list:
        comparison['systems_tested'].append(results.system_name)

        if results.equivalence_test.equivalence_confirmed:
            comparison['equivalence_confirmed'] += 1

        comparison['average_correlations'].append(results.equivalence_test.average_position_correlation)
        comparison['average_similarities'].append(results.equivalence_test.average_trajectory_similarity)
        comparison['average_period_matches'].append(results.equivalence_test.average_orbital_period_match)

        # Calculate success rate per system
        if results.system_name not in comparison['success_rates']:
            comparison['success_rates'][results.system_name] = {
                'total': 0,
                'successful': 0
            }

        comparison['success_rates'][results.system_name]['total'] += 1
        if results.equivalence_test.equivalence_confirmed:
            comparison['success_rates'][results.system_name]['successful'] += 1

    # Calculate overall statistics
    comparison['overall_success_rate'] = comparison['equivalence_confirmed'] / comparison['total_experiments']
    comparison['average_correlation'] = sum(comparison['average_correlations']) / len(comparison['average_correlations'])
    comparison['average_similarity'] = sum(comparison['average_similarities']) / len(comparison['average_similarities'])
    comparison['average_period_match'] = sum(comparison['average_period_matches']) / len(comparison['average_period_matches'])

    # Calculate system-specific success rates
    for system_name in comparison['success_rates']:
        stats = comparison['success_rates'][system_name]
        stats['success_rate'] = stats['successful'] / stats['total']

    return comparison


def generate_comparison_report(comparison: Dict[str, Any]) -> str:
    """Generate report comparing multiple experiment results.

    Args:
        comparison: Comparison analysis from compare_results()

    Returns:
        Formatted comparison report
    """
    if 'error' in comparison:
        return f"Error: {comparison['error']}"

    report = f"""
EXP-19 MULTIPLE EXPERIMENT COMPARISON REPORT
{'=' * 80}

OVERVIEW:
- Total experiments: {comparison['total_experiments']}
- Systems tested: {', '.join(comparison['systems_tested'])}
- Overall success rate: {comparison['overall_success_rate']:.1%}

AVERAGE METRICS ACROSS ALL EXPERIMENTS:
- Average position correlation: {comparison['average_correlation']:.6f}
- Average trajectory similarity: {comparison['average_similarity']:.6f}
- Average orbital period match: {comparison['average_period_match']:.6f}

SYSTEM-SPECIFIC SUCCESS RATES:
"""

    for system_name, stats in comparison['success_rates'].items():
        report += f"""
{system_name}:
  - Total runs: {stats['total']}
  - Successful: {stats['successful']}
  - Success rate: {stats['success_rate']:.1%}
"""

    # Overall conclusion
    if comparison['overall_success_rate'] == 1.0:
        report += """
🎉 PERFECT SUCCESS RATE!

All experiments confirmed orbital equivalence across all tested systems.
The hypothesis is strongly validated and ready for publication.
"""
    elif comparison['overall_success_rate'] >= 0.8:
        report += """
✅ STRONG EVIDENCE FOR EQUIVALENCE

Most experiments confirmed orbital equivalence, with strong evidence
supporting the hypothesis. Minor issues may require parameter refinement.
"""
    elif comparison['overall_success_rate'] >= 0.5:
        report += """
⚠️  MIXED RESULTS

Some experiments confirmed equivalence while others failed. This suggests
the hypothesis may be partially correct but requires significant refinement.
"""
    else:
        report += """
❌ HYPOTHESIS NOT SUPPORTED

Most experiments failed to confirm equivalence. The hypothesis requires
fundamental revision or may be incorrect.
"""

    report += f"""
{'=' * 80}
END OF COMPARISON REPORT
{'=' * 80}
"""

    return report


def export_visualization_data(results: EXP19_Results, output_dir: str = "visualization_data") -> Dict[str, str]:
    """Export data for visualization tools.

    Args:
        results: Complete experiment results
        output_dir: Directory to save visualization data

    Returns:
        Dictionary mapping data types to file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    exported_files = {}

    # Export trajectory data for 3D visualization
    trajectory_data = {
        'classical': results.classical_trajectories,
        'fractal': results.fractal_trajectories,
        'system_name': results.system_name,
        'timestamp': results.timestamp
    }

    trajectory_file = os.path.join(output_dir, f"trajectories_{results.system_name.replace(' ', '_')}.json")
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    exported_files['trajectories'] = trajectory_file

    # Export comparison metrics for plotting
    comparison_data = {
        'body_metrics': {name: comp.get_comparison_summary() for name, comp in results.comparisons.items()},
        'overall_metrics': {
            'position_correlation': results.equivalence_test.average_position_correlation,
            'trajectory_similarity': results.equivalence_test.average_trajectory_similarity,
            'orbital_period_match': results.equivalence_test.average_orbital_period_match
        },
        'system_name': results.system_name,
        'timestamp': results.timestamp
    }

    comparison_file = os.path.join(output_dir, f"comparison_metrics_{results.system_name.replace(' ', '_')}.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    exported_files['comparison_metrics'] = comparison_file

    # Export energy analysis data
    energy_file = os.path.join(output_dir, f"energy_analysis_{results.system_name.replace(' ', '_')}.json")
    with open(energy_file, 'w') as f:
        json.dump(results.energy_analysis, f, indent=2)
    exported_files['energy_analysis'] = energy_file

    print(f"Visualization data exported to {output_dir}/")
    return exported_files


def create_experiment_summary(results: EXP19_Results) -> Dict[str, Any]:
    """Create a concise summary of experiment results.

    Args:
        results: Complete experiment results

    Returns:
        Concise summary dictionary
    """
    summary = {
        'experiment_name': results.experiment_name,
        'system_name': results.system_name,
        'timestamp': results.timestamp,
        'equivalence_confirmed': results.equivalence_test.equivalence_confirmed,
        'overall_score': (results.equivalence_test.average_position_correlation +
                         results.equivalence_test.average_trajectory_similarity +
                         results.equivalence_test.average_orbital_period_match) / 3,
        'key_metrics': {
            'position_correlation': results.equivalence_test.average_position_correlation,
            'trajectory_similarity': results.equivalence_test.average_trajectory_similarity,
            'orbital_period_match': results.equivalence_test.average_orbital_period_match
        },
        'energy_conservation_equivalent': results.energy_analysis['energy_equivalence'],
        'perturbation_tested': results.perturbation_analysis is not None,
        'perturbation_equivalent': results.perturbation_analysis['perturbation_equivalence'] if results.perturbation_analysis else None,
        'validation_passed': all([
            results.validation_results['body_mapping_valid'],
            results.validation_results['mass_ratio_preserved'],
            results.validation_results['distance_mapping_valid']
        ])
    }

    return summary