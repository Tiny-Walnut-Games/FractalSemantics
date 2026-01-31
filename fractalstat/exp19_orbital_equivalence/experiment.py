"""
EXP-19: Orbital Equivalence - Main Experiment Orchestration

Implements the main experiment execution that tests whether orbital mechanics
and fractal cohesion mechanics produce equivalent predictions for celestial systems.

This module orchestrates the complete equivalence testing process:
1. System setup and calibration
2. Classical and fractal trajectory simulation
3. Comprehensive comparison and validation
4. Perturbation testing
5. Results analysis and reporting

Key Features:
- Automated testing of multiple orbital systems
- Calibration of fractal parameters to match classical gravity
- Statistical validation of equivalence
- Perturbation response testing
- Comprehensive reporting

Scientific Significance:
This experiment directly tests the hypothesis that Newtonian gravity
emerges from fractal topological interactions, potentially unifying
classical and quantum descriptions of gravity.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .entities import CelestialBody, OrbitalSystem, FractalBody, FractalOrbitalSystem
from .classical_mechanics import simulate_orbital_trajectory, validate_energy_conservation
from .fractal_mechanics import simulate_fractal_trajectory, calibrate_fractal_system, validate_fractal_classical_equivalence
from .comparison import TrajectoryComparison, OrbitalEquivalenceTest
from .systems import get_all_test_systems, validate_system_mapping
from .results import save_results


def run_orbital_equivalence_test(
    system_name: str = "Earth-Sun",
    simulation_time: float = 365 * 24 * 3600,  # 1 year in seconds
    time_steps: int = 1000,
    include_perturbation: bool = False,
    perturbation_time: float = 180 * 24 * 3600,  # Halfway through simulation
    G: float = 6.67430e-11
) -> OrbitalEquivalenceTest:
    """
    Run the orbital equivalence test between classical and fractal mechanics.

    This is the main experiment function that:
    1. Sets up the specified orbital system in both frameworks
    2. Calibrates fractal parameters to match classical gravity
    3. Runs trajectory simulations in both frameworks
    4. Compares predictions for equivalence
    5. Optionally tests perturbation responses

    Args:
        system_name: Name of the test system (e.g., "Earth-Sun", "Solar System")
        simulation_time: Total simulation time in seconds
        time_steps: Number of time steps for simulation
        include_perturbation: Whether to test perturbation response
        perturbation_time: When to introduce perturbation (seconds)
        G: Gravitational constant

    Returns:
        OrbitalEquivalenceTest containing all results and comparisons

    Raises:
        ValueError: If system name is invalid or simulation fails
    """
    print(f"Running orbital equivalence test for {system_name}...")
    print(f"Simulation time: {simulation_time / (24*3600):.1f} days")
    print(f"Time steps: {time_steps}")

    # Get test systems
    test_systems = get_all_test_systems()
    if system_name not in test_systems:
        available = list(test_systems.keys())
        raise ValueError(f"Unknown system: {system_name}. Available: {available}")

    classical_system, fractal_system = test_systems[system_name]

    # Validate system mapping
    mapping_validation = validate_system_mapping(classical_system, fractal_system)
    if not mapping_validation['body_mapping_valid']:
        print("Warning: System mapping issues detected:")
        for issue in mapping_validation['issues']:
            print(f"  - {issue}")

    # Calibrate fractal system to match classical gravity
    print("Calibrating fractal system parameters...")
    calibrated_fractal_system = calibrate_fractal_system(classical_system, fractal_system, G)
    print(f"Calibrated cohesion constant: {calibrated_fractal_system.cohesion_constant:.2e}")

    # Run classical simulation
    print("Running classical orbital simulation...")
    classical_trajectories = simulate_orbital_trajectory(
        classical_system, simulation_time, time_steps, G
    )

    # Validate classical energy conservation
    classical_energy_validation = validate_energy_conservation(classical_trajectories, G)
    print(f"Classical energy conservation: {'✓' if classical_energy_validation['conservation_valid'] else '✗'}")
    if not classical_energy_validation['conservation_valid']:
        print(f"  Energy drift: {classical_energy_validation['energy_drift']:.3f}%")

    # Run fractal simulation
    print("Running fractal orbital simulation...")
    fractal_trajectories = simulate_fractal_trajectory(
        calibrated_fractal_system, simulation_time, time_steps
    )

    # Create trajectory comparisons
    comparisons = {}
    for body_name in classical_trajectories:
        if body_name in fractal_trajectories:
            classical_data = classical_trajectories[body_name]
            fractal_data = fractal_trajectories[body_name]

            comparison = TrajectoryComparison(
                body_name=body_name,
                classical_positions=classical_data['positions'],
                fractal_positions=fractal_data['positions'],
                times=classical_data['times']
            )
            comparisons[body_name] = comparison

    # Create main equivalence test
    equivalence_test = OrbitalEquivalenceTest(
        system_name=system_name,
        classical_trajectories=classical_trajectories,
        fractal_trajectories=fractal_trajectories,
        comparisons=comparisons
    )

    # Test perturbation response if requested
    perturbed_comparisons = {}
    if include_perturbation:
        print("Testing perturbation response...")
        perturbed_test = run_perturbation_test(
            classical_system, calibrated_fractal_system,
            simulation_time, time_steps, perturbation_time, G
        )
        perturbed_comparisons = perturbed_test.comparisons

        # Analyze perturbation response equivalence
        perturbation_analysis = equivalence_test.analyze_perturbation_response(perturbed_comparisons)
        equivalence_test.perturbation_equivalence = perturbation_analysis['perturbation_equivalence']

    # Validate energy conservation equivalence
    classical_energies = {name: data['energies']['total'] for name, data in classical_trajectories.items()}
    fractal_energies = {name: data['energies']['total'] for name, data in fractal_trajectories.items()}
    energy_analysis = equivalence_test.validate_energy_conservation(classical_energies, fractal_energies)

    # Print results summary
    print("\n" + "="*60)
    print(f"ORBITAL EQUIVALENCE TEST RESULTS: {system_name}")
    print("="*60)
    print(f"Equivalence confirmed: {'✓ YES' if equivalence_test.equivalence_confirmed else '✗ NO'}")
    print(f"Average position correlation: {equivalence_test.average_position_correlation:.6f}")
    print(f"Average trajectory similarity: {equivalence_test.average_trajectory_similarity:.6f}")
    print(f"Average orbital period match: {equivalence_test.average_orbital_period_match:.6f}")

    if include_perturbation:
        print(f"Perturbation response equivalent: {'✓ YES' if equivalence_test.perturbation_equivalence else '✗ NO'}")

    print(f"Energy conservation equivalent: {'✓ YES' if energy_analysis['energy_equivalence'] else '✗ NO'}")

    if equivalence_test.equivalence_confirmed:
        print("\n🎉 BREAKTHROUGH CONFIRMED!")
        print("Orbital mechanics IS fractal mechanics under a different representation!")
        print("Newtonian gravity emerges from fractal hierarchical topology.")
    else:
        print("\n⚠️  EQUIVALENCE NOT CONFIRMED")
        print("Further investigation needed to understand discrepancies.")

    return equivalence_test


def run_perturbation_test(
    classical_system: OrbitalSystem,
    fractal_system: FractalOrbitalSystem,
    simulation_time: float,
    time_steps: int,
    perturbation_time: float,
    G: float = 6.67430e-11
) -> OrbitalEquivalenceTest:
    """
    Test how both frameworks respond to external perturbations.

    Introduces a massive rogue planet to test whether classical and fractal
    mechanics respond identically to external disturbances.

    Args:
        classical_system: Original classical system
        fractal_system: Original fractal system
        simulation_time: Total simulation time
        time_steps: Number of time steps
        perturbation_time: When to introduce perturbation
        G: Gravitational constant

    Returns:
        OrbitalEquivalenceTest with perturbed trajectories
    """
    print(f"Adding rogue planet perturbation at t={perturbation_time / (24*3600):.1f} days...")

    # Add perturbation to both systems
    perturbed_classical, perturbed_fractal = add_rogue_planet_perturbation(
        classical_system, fractal_system, perturbation_time
    )

    # Run perturbed simulations
    perturbed_classical_traj = simulate_orbital_trajectory(
        perturbed_classical, simulation_time - perturbation_time, time_steps // 2, G
    )
    perturbed_fractal_traj = simulate_fractal_trajectory(
        perturbed_fractal, simulation_time - perturbation_time, time_steps // 2
    )

    # Combine original and perturbed trajectories
    combined_classical = {}
    combined_fractal = {}

    for body_name in perturbed_classical_traj.keys():
        if body_name in perturbed_fractal_traj and body_name != "Rogue Planet":
            # Get original trajectories
            orig_classical = classical_system.bodies[0].position  # Simplified
            orig_fractal = fractal_system.bodies[0].tree_address  # Simplified

            # Combine trajectories
            combined_classical[body_name] = {
                'positions': orig_classical + perturbed_classical_traj[body_name]['positions'],
                'times': list(range(len(orig_classical) + len(perturbed_classical_traj[body_name]['positions'])))
            }
            combined_fractal[body_name] = {
                'positions': orig_fractal + perturbed_fractal_traj[body_name]['positions'],
                'times': list(range(len(orig_fractal) + len(perturbed_fractal_traj[body_name]['positions'])))
            }

    # Create comparisons for perturbed trajectories
    perturbed_comparisons = {}
    for body_name in combined_classical:
        if body_name in combined_fractal:
            comparison = TrajectoryComparison(
                body_name=body_name,
                classical_positions=combined_classical[body_name]['positions'],
                fractal_positions=combined_fractal[body_name]['positions'],
                times=combined_classical[body_name]['times']
            )
            perturbed_comparisons[body_name] = comparison

    return OrbitalEquivalenceTest(
        system_name=f"{classical_system.name} (Perturbed)",
        classical_trajectories=combined_classical,
        fractal_trajectories=combined_fractal,
        comparisons=perturbed_comparisons
    )


def add_rogue_planet_perturbation(
    classical_system: OrbitalSystem,
    fractal_system: FractalOrbitalSystem,
    perturbation_time: float
) -> Tuple[OrbitalSystem, FractalOrbitalSystem]:
    """
    Add a massive rogue planet to both systems to test perturbation response.

    Args:
        classical_system: Original classical system
        fractal_system: Original fractal system
        perturbation_time: Time when perturbation occurs

    Returns:
        Tuple of perturbed (classical_system, fractal_system)
    """
    import numpy as np

    # Add massive rogue planet to classical system
    rogue_mass = 2.0e27  # ~0.1 solar masses
    rogue_distance = 5.0e11  # 3-4 AU
    rogue_velocity = 1.0e4  # 10 km/s

    rogue_classical = CelestialBody(
        name="Rogue Planet",
        mass=rogue_mass,
        radius=5.0e7,
        position=np.array([rogue_distance, 0.0, 0.0]),
        velocity=np.array([0.0, rogue_velocity, 0.0])
    )

    perturbed_classical = OrbitalSystem(
        name=f"{classical_system.name} (Perturbed)",
        bodies=classical_system.bodies + [rogue_classical],
        central_body=classical_system.central_body
    )

    # Add corresponding fractal rogue planet
    rogue_fractal = FractalBody(
        name="Rogue Planet",
        fractal_density=0.001,  # 0.1% of solar density
        hierarchical_depth=5,   # Distant hierarchy
        tree_address=[99]       # External branch
    )

    perturbed_fractal = FractalOrbitalSystem(
        name=f"{fractal_system.name} (Perturbed)",
        bodies=fractal_system.bodies + [rogue_fractal],
        central_body=fractal_system.central_body,
        max_hierarchy_depth=max(fractal_system.max_hierarchy_depth, 6),
        cohesion_constant=fractal_system.cohesion_constant
    )

    return perturbed_classical, perturbed_fractal


def run_comprehensive_equivalence_suite(
    systems_to_test: Optional[List[str]] = None,
    simulation_time: float = 365 * 24 * 3600,
    time_steps: int = 1000,
    include_perturbations: bool = True
) -> Dict[str, OrbitalEquivalenceTest]:
    """
    Run comprehensive equivalence testing across multiple orbital systems.

    Tests the hypothesis across different types of celestial systems to
    ensure the equivalence is general and not specific to one configuration.

    Args:
        systems_to_test: List of system names to test (defaults to all)
        simulation_time: Simulation time for each system
        time_steps: Time steps for each simulation
        include_perturbations: Whether to test perturbation responses

    Returns:
        Dictionary mapping system names to test results
    """
    if systems_to_test is None:
        systems_to_test = list(get_all_test_systems().keys())

    print(f"Running comprehensive equivalence suite for {len(systems_to_test)} systems...")
    print(f"Systems: {', '.join(systems_to_test)}")

    results = {}

    for system_name in systems_to_test:
        print(f"\n{'='*80}")
        print(f"TESTING SYSTEM: {system_name}")
        print('='*80)

        try:
            # Determine perturbation time based on system
            if system_name == "Exoplanetary System":
                perturbation_time = 0.5 * simulation_time  # Hot Jupiter system
            else:
                perturbation_time = 180 * 24 * 3600  # Standard perturbation time

            result = run_orbital_equivalence_test(
                system_name=system_name,
                simulation_time=simulation_time,
                time_steps=time_steps,
                include_perturbation=include_perturbations,
                perturbation_time=perturbation_time
            )

            results[system_name] = result

            # Print system-specific summary
            summary = result.get_system_summary()
            print(f"\n{system_name} Summary:")
            print(f"  Equivalence: {'✓' if summary['equivalence_confirmed'] else '✗'}")
            print(f"  Equivalent bodies: {summary['equivalent_bodies']}/{summary['total_bodies']}")
            print(f"  Average correlation: {summary['average_position_correlation']:.6f}")

        except Exception as e:
            print(f"Error testing {system_name}: {e}")
            results[system_name] = None

    # Print overall suite summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUITE SUMMARY")
    print('='*80)

    successful_tests = sum(1 for r in results.values() if r is not None and r.equivalence_confirmed)
    total_tests = len(results)

    print(f"Total systems tested: {total_tests}")
    print(f"Successful equivalence tests: {successful_tests}")
    print(f"Success rate: {(successful_tests / total_tests * 100):.1f}%")

    if successful_tests == total_tests:
        print("\n🎉 ALL SYSTEMS CONFIRM EQUIVALENCE!")
        print("The hypothesis is strongly supported across diverse celestial configurations.")
    elif successful_tests > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {successful_tests}/{total_tests} systems confirmed equivalence")
        print("Further investigation needed for systems that failed.")
    else:
        print("\n❌ NO SYSTEMS CONFIRMED EQUIVALENCE")
        print("The hypothesis requires significant revision.")

    return results


def validate_gravitational_constant_emergence(
    classical_system: OrbitalSystem,
    fractal_system: FractalOrbitalSystem,
    G: float = 6.67430e-11
) -> Dict[str, Any]:
    """
    Validate whether Newton's gravitational constant emerges from fractal parameters.

    This is the key test: can we derive G from purely fractal topological parameters?

    Args:
        classical_system: Reference classical system
        fractal_system: Corresponding fractal system
        G: Known gravitational constant

    Returns:
        Analysis of gravitational constant emergence
    """
    from .fractal_mechanics import analyze_fractal_gravitational_constant

    analysis = analyze_fractal_gravitational_constant(classical_system, fractal_system, G)

    print(f"\nGRAVITATIONAL CONSTANT EMERGENCE ANALYSIS:")
    print(f"Known G: {G:.6e} m³ kg⁻¹ s⁻²")
    print(f"Derived G: {analysis['derived_g']:.6e} m³ kg⁻¹ s⁻²")
    print(f"Ratio: {analysis['ratio']:.6f}")
    print(f"Derivable: {'✓ YES' if analysis['derivable'] else '✗ NO'}")

    if analysis['derivable']:
        print("\n🎉 GRAVITATIONAL CONSTANT EMERGES FROM FRACTAL TOPOLOGY!")
        print("This is the smoking gun evidence for the hypothesis.")
    else:
        print("\n❌ GRAVITATIONAL CONSTANT DOES NOT EMERGE")
        print("The hypothesis requires fundamental revision.")

    return analysis


def run_quick_validation_test() -> OrbitalEquivalenceTest:
    """
    Run a quick validation test with the Earth-Sun system.

    Provides rapid feedback on whether the equivalence hypothesis holds
    for the most fundamental orbital configuration.

    Returns:
        Test results for quick validation
    """
    print("Running quick validation test (Earth-Sun system)...")
    return run_orbital_equivalence_test(
        system_name="Earth-Sun",
        simulation_time=365 * 24 * 3600,  # 1 year
        time_steps=500,  # Faster simulation
        include_perturbation=False  # Skip for speed
    )