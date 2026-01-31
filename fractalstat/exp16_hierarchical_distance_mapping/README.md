# EXP-16: Hierarchical Distance to Euclidean Distance Mapping

## Overview

EXP-16 tests whether hierarchical distance (discrete tree hops) maps to Euclidean distance (continuous spatial distance) through fractal embedding strategies. This experiment is crucial for understanding how discrete fractal physics connects to continuous Newtonian physics.

## Core Hypothesis

When fractal hierarchies are embedded in Euclidean space, hierarchical distance `d_h` (tree hops) relates to Euclidean distance `r` through a power-law:

```
r ∝ d_h^exponent
```

This mapping explains why fractal gravity produces Newtonian-like inverse-square forces.

## Experiment Phases

### 1. Fractal Embedding Strategies
Three different embedding strategies are tested:

- **Exponential Embedding**: Nodes placed at exponentially increasing distances from root
- **Spherical Embedding**: Nodes placed on concentric spheres (shell-based)
- **Recursive Embedding**: Recursive space division along coordinate axes

### 2. Distance Measurement
For each embedding strategy:
- Measure hierarchical distance (tree hops) between random node pairs
- Measure Euclidean distance (spatial distance) between embedded positions
- Analyze correlation between the two distance measures

### 3. Power-Law Analysis
- Fit power-law relationship: `e_distance = coefficient * h_distance^exponent`
- Calculate correlation coefficient for the fit
- Determine optimal exponent value

### 4. Force Scaling Validation
- Compare hierarchical force scaling (inverse-square on tree distance)
- Compare Euclidean force scaling (Newtonian inverse-square)
- Validate consistency between discrete and continuous approaches

## Success Criteria

- **Distance correlation > 0.95** for best embedding strategy
- **Power-law exponent found** in range 1 ≤ exponent ≤ 2
- **Force correlation > 0.90** between hierarchical and Euclidean
- **Optimal embedding type identified**

## Key Components

### Core Entities

#### `EmbeddedFractalHierarchy`
Represents a fractal hierarchy embedded in Euclidean space with:
- Original fractal hierarchy
- Embedding type (strategy used)
- Node positions in 3D space
- Methods to compute both distance types

#### `EmbeddingStrategy` (Base Class)
Abstract base class for embedding strategies with:
- Strategy name and description
- `embed_hierarchy()` method to place nodes in space

#### Concrete Embedding Strategies

**`ExponentialEmbedding`**
- Places nodes at exponential distance from parent
- Formula: `radius = scale_factor * (2.0^depth)`
- Distributes children in circles around parents

**`SphericalEmbedding`**
- Places nodes on concentric spherical shells
- Radius increases linearly with depth
- Uses golden spiral distribution for even spacing

**`RecursiveEmbedding`**
- Recursive space partitioning along coordinate axes
- Each branching splits space into 6 directions
- Maintains hierarchical structure in spatial layout

#### `DistancePair`
Represents a pair of nodes with both distance measurements:
- Hierarchical distance (tree hops)
- Euclidean distance (spatial)
- Distance ratio calculation

#### `DistanceMappingAnalysis`
Complete analysis of distance mapping for an embedded hierarchy:
- Power-law exponent and coefficient
- Correlation coefficient
- Overall mapping quality score

#### `ForceScalingValidation`
Validation of force scaling consistency:
- Hierarchical vs Euclidean force correlation
- Scaling consistency analysis

### Experiment Logic

#### `test_embedding_strategy()`
Tests a specific embedding strategy with given parameters:
- Builds fractal hierarchy
- Embeds using the strategy
- Analyzes distance mapping
- Validates force scaling
- Returns quality metrics

#### `run_exp16_distance_mapping_experiment()`
Main experiment runner that:
- Tests all embedding strategies
- Optimizes over scale factors
- Finds best overall strategy
- Validates success criteria
- Returns complete results

## Usage

### Basic Usage

```python
from fractalstat.exp16_hierarchical_distance_mapping import run_exp16_distance_mapping_experiment

# Run with default parameters
results = run_exp16_distance_mapping_experiment()

print(f"Best embedding: {results.best_embedding_strategy}")
print(f"Optimal exponent: {results.optimal_exponent}")
print(f"Experiment success: {results.experiment_success}")
```

### Custom Parameters

```python
# Run with custom parameters
results = run_exp16_distance_mapping_experiment(
    hierarchy_depth=6,           # Deeper hierarchy
    branching_factor=4,          # More branching
    scale_factors=[0.5, 1.0, 2.0],  # Multiple scale factors
    distance_samples=2000        # More samples for accuracy
)
```

### Individual Strategy Testing

```python
from fractalstat.exp16_hierarchical_distance_mapping import (
    ExponentialEmbedding,
    test_embedding_strategy
)

# Test specific strategy
strategy = ExponentialEmbedding()
result = test_embedding_strategy(
    strategy=strategy,
    hierarchy_depth=5,
    branching_factor=3,
    scale_factor=1.0,
    distance_samples=1000
)

print(f"Strategy: {result.strategy_name}")
print(f"Distance correlation: {result.distance_correlation:.4f}")
print(f"Force correlation: {result.force_correlation:.4f}")
print(f"Overall quality: {result.overall_quality:.4f}")
```

## Command Line Interface

### Quick Test
```bash
cd fractalstat
python exp16_hierarchical_distance_mapping.py --quick
```

### Full Test
```bash
cd fractalstat
python exp16_hierarchical_distance_mapping.py --full
```

### Configuration
Create a configuration file to customize parameters:

```toml
[EXP-16]
hierarchy_depth = 6
branching_factor = 4
scale_factors = [0.5, 1.0, 1.5, 2.0]
distance_samples = 2000
```

## Results Format

Results are saved as JSON with the following structure:

```json
{
  "experiment": "EXP-16",
  "test_type": "Hierarchical Distance to Euclidean Distance Mapping",
  "start_time": "2025-12-07T23:06:05.123456+00:00",
  "end_time": "2025-12-07T23:06:45.654321+00:00",
  "total_duration_seconds": 40.53,
  "parameters": {
    "hierarchy_depth": 5,
    "branching_factor": 3,
    "distance_samples": 1000
  },
  "embedding_results": {
    "Exponential": {
      "strategy_name": "Exponential",
      "distance_correlation": 0.945678,
      "force_correlation": 0.923456,
      "exponent_in_range": true,
      "overall_quality": 0.956432,
      "distance_analysis": {
        "power_law_exponent": 1.456789,
        "power_law_coefficient": 0.876543,
        "correlation_coefficient": 0.945678,
        "mapping_quality": 0.945678
      },
      "force_validation": {
        "force_correlation": 0.923456,
        "scaling_consistency": 0.923456
      }
    }
  },
  "analysis": {
    "best_embedding_strategy": "Exponential",
    "optimal_exponent": 1.456789,
    "distance_mapping_success": true,
    "force_scaling_consistent": true,
    "experiment_success": true
  }
}
```

## Dependencies

### Required Modules
- `fractalstat.exp13_fractal_gravity`: For `FractalHierarchy` and `FractalNode`
- `numpy`: For numerical computations and linear algebra
- `scipy`: For statistical analysis and correlation calculations

### Optional Dependencies
- `matplotlib`: For visualization of embedding results
- `seaborn`: For enhanced plotting capabilities

## Performance Considerations

### Memory Usage
- Embedding strategies scale with hierarchy size
- Distance measurement scales with `O(n²)` for all pairs or `O(n)` for sampling
- Large hierarchies (>1000 nodes) may require sampling

### Computation Time
- Embedding time: O(n) where n is number of nodes
- Distance measurement: O(k) where k is number of samples
- Power-law fitting: O(k) for linear regression on log scales

### Optimization Tips
- Use `--quick` mode for development and testing
- Adjust `distance_samples` based on available memory
- Consider hierarchy depth limits for very large systems

## Integration with Other Experiments

### EXP-13: Fractal Gravity
- Uses `FractalHierarchy` and `FractalNode` from EXP-13
- Validates that fractal hierarchies can produce Newtonian-like forces

### EXP-20: Vector Field Derivation
- Provides foundation for understanding force scaling
- Distance mapping validates vector field approaches

### EXP-17: Thermodynamic Validation
- Distance mapping helps explain energy non-conservation
- Hierarchical vs Euclidean distance affects energy calculations

## Troubleshooting

### Common Issues

**Import Errors**
```python
# Ensure EXP-13 is available
from fractalstat.exp13_fractal_gravity import FractalHierarchy
```

**Memory Errors**
```python
# Reduce hierarchy size or sampling
results = run_exp16_distance_mapping_experiment(
    hierarchy_depth=4,  # Smaller hierarchy
    distance_samples=500  # Fewer samples
)
```

**Poor Correlation**
```python
# Try different embedding strategies or scale factors
results = run_exp16_distance_mapping_experiment(
    scale_factors=[0.5, 1.0, 1.5, 2.0, 3.0]  # More scale factors
)
```

### Debug Mode
Enable debug output by setting environment variable:
```bash
export EXP16_DEBUG=1
python exp16_hierarchical_distance_mapping.py
```

## Future Enhancements

### Planned Features
- Additional embedding strategies (spiral, lattice-based)
- Visualization tools for embedding results
- Performance optimization for large hierarchies
- Integration with machine learning for optimal embedding discovery

### Research Directions
- Optimal embedding for specific physical systems
- Relationship between embedding strategy and physical laws
- Multi-scale embedding for hierarchical systems
- Dynamic embedding for evolving hierarchies

## References

- Fractal Physics Theory: Discrete vs Continuous Approaches
- Hierarchical Embedding in Euclidean Space
- Power-Law Relationships in Complex Systems
- Force Scaling in Fractal Systems