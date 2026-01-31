# EXP-15: Topological Conservation Laws

## Overview

EXP-15 tests whether fractal systems conserve topology (hierarchical structure, connectivity, branching patterns) rather than classical energy and momentum. This experiment represents a fundamental test of whether fractal physics operates under different conservation laws than Newtonian physics.

## Core Hypothesis

**In fractal physics, topology is the conserved quantity, not energy.**

- Classical Newtonian mechanics conserves energy but not topology
- Fractal mechanics conserves topology but not energy
- This represents a fundamental ontological difference between the two frameworks

## Experiment Design

### Phases

1. **Define Topological Invariants**: Establish complete set of topological properties that should be conserved
2. **Orbital Dynamics Simulation**: Run orbital mechanics with topological tracking
3. **Classical Conservation Analysis**: Compare against Newtonian conservation laws
4. **Validation**: Prove topology conserved while energy is not

### Success Criteria

- **Topology Conservation**: 100% stability over 1-year orbital simulation
- **Energy Non-Conservation**: Classical energy shows measurable drift
- **Invariant Preservation**: Node count, depth, connectivity remain constant
- **Zero Collisions**: Address collision count remains zero
- **Entropy Stability**: Structure entropy stays constant

## Key Components

### Topological Invariants

The experiment tracks these conserved quantities:

- **Total Nodes**: Count of entities in the system
- **Max Hierarchical Depth**: Deepest level in fractal hierarchy
- **Branching Distribution**: Distribution of branching factors across entities
- **Connectivity Matrix**: Hash of parent-child relationships
- **Address Collision Count**: Number of address collisions (should be zero)
- **Structure Entropy**: Shannon entropy of branching distribution
- **Fractal Dimension**: Approximation of system complexity

### Classical Conservation Laws

For comparison, the experiment also tracks:

- **Energy Conservation**: Kinetic + potential energy over time
- **Momentum Conservation**: Linear momentum magnitude
- **Angular Momentum Conservation**: Angular momentum magnitude

## Usage

### Basic Usage

```python
from fractalstat.exp15_topological_conservation import TopologicalConservationExperiment

# Create experiment with default settings
experiment = TopologicalConservationExperiment()

# Run the experiment
results = experiment.run()

# Check results
if results["analysis"]["fractal_physics_validated"]:
    print("Fractal physics validated!")
else:
    print("Further investigation needed.")
```

### Advanced Configuration

```python
# Configure specific systems and approaches
experiment = TopologicalConservationExperiment(
    systems_to_test=["Earth-Sun", "Moon-Earth"],
    approaches_to_test=["Branching Vector (Ratio)", "Vector Field (Gradient)"]
)

results = experiment.run()
```

### Direct Function Usage

```python
from fractalstat.exp15_topological_conservation import (
    compute_topological_invariants,
    compute_classical_conservation,
    integrate_orbit_with_topological_tracking
)

# Compute topological invariants for a system
invariants = compute_topological_invariants(entities, timestamp)

# Analyze classical conservation
classical_analysis = compute_classical_conservation(trajectory, central_mass)

# Run orbital simulation with topological tracking
trajectory, topological_analysis = integrate_orbit_with_topological_tracking(
    orbiting_entity, central_entity, vector_approach, scalar_magnitude,
    time_span=365.25 * 24 * 3600,  # 1 year
    time_steps=1000
)
```

## Data Structures

### TopologicalInvariants

```python
@dataclass
class TopologicalInvariants:
    timestamp: float
    total_nodes: int
    max_hierarchical_depth: int
    branching_distribution: Dict[int, int]
    connectivity_matrix_hash: str
    address_collision_count: int
    structure_entropy: float
    fractal_dimension: float
```

### TopologicalConservationAnalysis

```python
@dataclass
class TopologicalConservationAnalysis:
    reference_measurement: TopologicalConservationMeasurement
    all_measurements: List[TopologicalConservationMeasurement]
    node_conservation_rate: float
    depth_conservation_rate: float
    connectivity_conservation_rate: float
    collision_conservation_rate: float
    entropy_conservation_rate: float
    topology_fully_conserved: bool
```

### ClassicalConservationAnalysis

```python
@dataclass
class ClassicalConservationAnalysis:
    times: List[float]
    energies: List[float]
    momenta: List[float]
    angular_momenta: List[float]
    energy_conservation_rate: float
    momentum_conservation_rate: float
    angular_momentum_conservation_rate: float
    classical_conservation_violated: bool
```

## Results Format

The experiment returns a comprehensive results dictionary:

```python
{
    "experiment": "EXP-15",
    "test_type": "Topological Conservation Laws",
    "start_time": "2025-12-07T23:06:05.123456Z",
    "end_time": "2025-12-07T23:07:15.654321Z",
    "total_duration_seconds": 70.53,
    "systems_tested": ["Earth-Sun"],
    "approaches_tested": ["Branching Vector (Ratio)"],
    "conservation_results": {
        "Earth-Sun": {
            "Branching Vector (Ratio)": {
                "system_name": "Earth-Sun",
                "approach_name": "Branching Vector (Ratio)",
                "integration_time": 12.345678,
                "topology_conserved": True,
                "classical_energy_not_conserved": True,
                "fundamental_difference_demonstrated": True,
                "topological_analysis": {
                    "node_conservation_rate": 1.0,
                    "depth_conservation_rate": 1.0,
                    "connectivity_conservation_rate": 1.0,
                    "collision_conservation_rate": 1.0,
                    "entropy_conservation_rate": 1.0,
                    "topology_fully_conserved": True
                },
                "classical_analysis": {
                    "energy_conservation_rate": 0.923456,
                    "momentum_conservation_rate": 0.999999,
                    "angular_momentum_conservation_rate": 0.999998,
                    "classical_conservation_violated": True
                }
            }
        }
    },
    "analysis": {
        "topology_conservation_confirmed": True,
        "classical_energy_nonconservation_confirmed": True,
        "fractal_physics_validated": True
    }
}
```

## CLI Usage

```bash
# Run with default settings
python -m fractalstat.exp15_topological_conservation

# Run quick test
python -m fractalstat.exp15_topological_conservation --quick

# Run full test (if configured)
python -m fractalstat.exp15_topological_conservation --full
```

## Dependencies

- `numpy`: For numerical computations
- `fractalstat.exp20_vector_field_derivation`: For orbital mechanics
- Standard library: `json`, `time`, `datetime`, `pathlib`, `dataclasses`

## Integration with Other Experiments

EXP-15 builds on:

- **EXP-20**: Uses vector field derivation for orbital mechanics
- **EXP-17**: Explains why energy non-conservation was observed
- **EXP-19**: Provides orbital equivalence framework

## Scientific Significance

This experiment addresses a fundamental question in physics:

**What are the true conserved quantities in nature?**

If successful, EXP-15 would demonstrate that:

1. **Topology is fundamental**: Hierarchical structure is preserved over time
2. **Energy is emergent**: Classical energy conservation is an approximation
3. **Fractal physics is valid**: Different ontological framework from Newtonian physics
4. **New conservation laws**: Topological invariants replace classical conservation

## Performance Characteristics

- **Computation Time**: ~1-2 minutes for full orbital simulation
- **Memory Usage**: Low (primarily tracking invariants, not full simulation state)
- **Accuracy**: High precision tracking of topological properties
- **Scalability**: Can be extended to multi-body systems

## Future Extensions

1. **Multi-body systems**: Extend to solar system scale simulations
2. **Quantum effects**: Incorporate quantum mechanical considerations
3. **Relativistic effects**: Include general relativity corrections
4. **Cosmological scales**: Apply to galaxy and universe-scale structures
5. **Biological systems**: Test topological conservation in biological hierarchies

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure EXP-20 dependencies are available
2. **Numerical precision**: Topological conservation requires high precision
3. **Integration time**: Longer simulations may require more time steps
4. **Memory usage**: Large systems may require memory optimization

### Validation

The experiment includes built-in validation:

- **Cross-checks**: Multiple conservation metrics
- **Reference points**: Initial state as conservation reference
- **Error bounds**: Tolerance for numerical precision
- **Statistical analysis**: Conservation rates over time

## Contributing

To extend this experiment:

1. Add new topological invariants to `TopologicalInvariants`
2. Implement new vector field approaches in `VectorFieldApproach`
3. Extend analysis functions for new conservation laws
4. Add new test systems and configurations

## License

This experiment is part of the FractalSemantics project and follows the same licensing terms.