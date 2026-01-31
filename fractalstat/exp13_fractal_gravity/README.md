# EXP-13: Fractal Gravity

## Overview

EXP-13 provides comprehensive fractal gravity testing capabilities for evaluating whether fractal entities naturally create gravitational cohesion without falloff, and whether injecting falloff produces consistent weakening across all element types.

This experiment tests the fundamental hypothesis that natural cohesion depends ONLY on hierarchical relationship (constant, no falloff), and that falloff injection produces identical mathematical patterns across all elements.

## Core Postulates

### 1. Fractal Cohesion Without Falloff
- Cohesion remains constant across hierarchical levels
- No distance-based weakening in pure fractal structures
- Hierarchical relationship determines interaction strength

### 2. Elements as Fractal Constructs
- Mass equals fractal density (hierarchical complexity)
- Atomic properties correlate with fractal structure complexity
- Element-specific cohesion magnitudes based on fractal properties

### 3. Universal Interaction Mechanism
- Same falloff pattern for all elements when applied
- Universal mathematical behavior across different materials
- Consistent response to external falloff injection

### 4. Hierarchical Distance is Fundamental
- Topology, not space, determines interactions
- Tree-based distance calculations
- Lowest common ancestor determines relationship strength

## Key Concepts

### Hierarchical Distance
Unlike Euclidean distance, hierarchical distance measures the number of hops through a tree structure to reach the lowest common ancestor:

```
Example Hierarchy:
Level 0: Root
Level 1: [0], [1], [2]
Level 2: [0,0], [0,1], [1,0], [1,1], [2,0], [2,1]

Node A: [0,1] (depth 2)
Node B: [1,0] (depth 2)
Common ancestor: Root (depth 0)
Hierarchical distance: (2-0) + (2-0) = 4
```

### Natural Cohesion
Cohesion that exists without any falloff mechanism applied:
- Depends only on hierarchical relationship
- Should be constant across all hierarchical distances
- Represents the fundamental fractal property

### Falloff Cohesion
Cohesion with artificial falloff applied:
- Tests universal falloff mechanism
- Should follow consistent mathematical patterns
- Validates inverse-square law application

## API Reference

### FractalGravityExperiment

Main experiment runner for fractal gravity testing.

```python
from fractalstat.exp13_fractal_gravity import FractalGravityExperiment

# Initialize experiment
experiment = FractalGravityExperiment(
    elements_to_test=["gold", "nickel", "copper"],  # Elements to test
    max_hierarchy_depth=5,                          # Tree depth
    interaction_samples=5000                        # Node pair samples
)

# Run experiment
results = experiment.run()

# Save results
from fractalstat.exp13_fractal_gravity import save_results
save_results(results)
```

#### Parameters

- **elements_to_test** (List[str]): List of element types to test (default: ["gold", "nickel", "copper", "iron", "silver"])
- **max_hierarchy_depth** (int): Maximum depth of fractal tree to build (default: 5)
- **interaction_samples** (int): Number of random node pairs to test per element (default: 5000)

#### Methods

- **run()** → EXP13v2_GravityTestResults: Execute the complete fractal gravity experiment
- **run_hierarchical_gravity_test_for_element(element_type: str)** → ElementGravityResults: Test single element
- **analyze_fractal_no_falloff()** → bool: Analyze natural cohesion flatness
- **analyze_universal_falloff_mechanism()** → bool: Analyze falloff pattern consistency
- **analyze_mass_fractal_correlation()** → float: Analyze element property correlations

### FractalNode

A node in a pure fractal hierarchy (no spatial coordinates).

```python
from fractalstat.exp13_fractal_gravity import FractalNode

# Create a node
node = FractalNode(
    element="gold",
    hierarchical_depth=3,
    tree_address=[0, 2, 1]
)

print(f"Node: {node}")
print(f"Element: {node.element}")
print(f"Depth: {node.hierarchical_depth}")
print(f"Address: {node.tree_address}")
```

#### Fields

- **element** (str): Element type (gold, nickel, etc.)
- **hierarchical_depth** (int): Depth in fractal tree (0 = root, 1 = child, etc.)
- **tree_address** (List[int]): Address in tree (e.g., [0, 2, 1] = grandchild of second child of root)

### FractalHierarchy

A pure fractal tree structure for a single element type.

```python
from fractalstat.exp13_fractal_gravity import FractalHierarchy

# Build hierarchy
hierarchy = FractalHierarchy.build(
    element_type="gold",
    max_depth=4,
    branching_factor=3
)

# Get all nodes
all_nodes = hierarchy.get_all_nodes()
print(f"Total nodes: {len(all_nodes)}")

# Calculate distance between nodes
node_a = all_nodes[0]
node_b = all_nodes[10]
distance = hierarchy.get_hierarchical_distance(node_a, node_b)
print(f"Hierarchical distance: {distance}")
```

#### Methods

- **build(element_type: str, max_depth: int, branching_factor: int = 3)** → FractalHierarchy: Build complete fractal tree
- **get_all_nodes()** → List[FractalNode]: Get all nodes in hierarchy
- **get_hierarchical_distance(node_a: FractalNode, node_b: FractalNode)** → int: Calculate hierarchical distance

### ElementGravityResults

Results for a single element's gravitational behavior.

```python
from fractalstat.exp13_fractal_gravity import ElementGravityResults

# Access results
result = ElementGravityResults(
    element="gold",
    total_measurements=5000,
    cohesion_by_hierarchical_distance={...},
    element_fractal_density=0.95
)

print(f"Element: {result.element}")
print(f"Total measurements: {result.total_measurements}")
print(f"Natural cohesion flatness: {result.natural_cohesion_flatness}")
print(f"Falloff pattern consistency: {result.falloff_pattern_consistency}")
print(f"Element fractal density: {result.element_fractal_density}")
```

#### Fields

- **element** (str): Element name
- **total_measurements** (int): Number of cohesion measurements
- **cohesion_by_hierarchical_distance** (Dict[int, Dict[str, Any]]): Cohesion data grouped by distance
- **element_fractal_density** (float): Derived fractal density property
- **natural_cohesion_flatness** (float): How constant natural cohesion is across distances (0.0-1.0)
- **falloff_pattern_consistency** (float): How well falloff follows inverse-square pattern (0.0-1.0)

### EXP13v2_GravityTestResults

Complete results from fractal gravity test.

```python
from fractalstat.exp13_fractal_gravity import EXP13v2_GravityTestResults

# Access complete results
results = EXP13v2_GravityTestResults(
    start_time="2025-01-01T12:00:00Z",
    end_time="2025-01-01T12:30:00Z",
    total_duration_seconds=1800.0,
    elements_tested=["gold", "nickel"],
    element_results={...},
    measurements=[...],
    fractal_no_falloff_confirmed=True,
    universal_falloff_mechanism=True,
    mass_fractal_density_correlation=0.85
)

print(f"Start time: {results.start_time}")
print(f"End time: {results.end_time}")
print(f"Duration: {results.total_duration_seconds}s")
print(f"Elements tested: {results.elements_tested}")
print(f"Fractal no falloff confirmed: {results.fractal_no_falloff_confirmed}")
print(f"Universal falloff mechanism: {results.universal_falloff_mechanism}")
print(f"Mass-fractal correlation: {results.mass_fractal_density_correlation}")
```

#### Fields

- **start_time** (str): Experiment start timestamp (ISO format)
- **end_time** (str): Experiment end timestamp (ISO format)
- **total_duration_seconds** (float): Total execution time
- **elements_tested** (List[str]): Elements that were tested
- **element_results** (Dict[str, ElementGravityResults]): Results for each element
- **measurements** (List[HierarchicalCohesionMeasurement]): All individual measurements
- **fractal_no_falloff_confirmed** (bool): Whether natural cohesion is constant across hierarchy
- **universal_falloff_mechanism** (bool): Whether falloff follows same pattern for all elements
- **mass_fractal_density_correlation** (float): Correlation between mass and fractal properties

## Usage Examples

### Basic Fractal Gravity Test

```python
from fractalstat.exp13_fractal_gravity import FractalGravityExperiment

# Quick test with default settings
experiment = FractalGravityExperiment()
results = experiment.run()

print(f"Test completed: {results.fractal_no_falloff_confirmed}")
print(f"Universal falloff: {results.universal_falloff_mechanism}")
```

### Custom Element Testing

```python
from fractalstat.exp13_fractal_gravity import FractalGravityExperiment

# Test specific elements with custom parameters
experiment = FractalGravityExperiment(
    elements_to_test=["gold", "silver", "copper"],
    max_hierarchy_depth=6,
    interaction_samples=10000
)

results = experiment.run()

# Analyze individual element results
for element, element_result in results.element_results.items():
    print(f"{element}:")
    print(f"  Natural cohesion flatness: {element_result.natural_cohesion_flatness:.4f}")
    print(f"  Falloff pattern consistency: {element_result.falloff_pattern_consistency:.4f}")
    print(f"  Fractal density: {element_result.element_fractal_density:.4f}")
```

### Hierarchical Structure Analysis

```python
from fractalstat.exp13_fractal_gravity import FractalHierarchy, FractalNode

# Build and analyze hierarchy
hierarchy = FractalHierarchy.build(
    element_type="gold",
    max_depth=4,
    branching_factor=3
)

all_nodes = hierarchy.get_all_nodes()
print(f"Total nodes: {len(all_nodes)}")

# Analyze node distribution by depth
nodes_by_depth = {}
for node in all_nodes:
    depth = node.hierarchical_depth
    if depth not in nodes_by_depth:
        nodes_by_depth[depth] = 0
    nodes_by_depth[depth] += 1

for depth, count in sorted(nodes_by_depth.items()):
    print(f"Depth {depth}: {count} nodes")
```

### Cohesion Analysis

```python
from fractalstat.exp13_fractal_gravity import (
    compute_natural_cohesion,
    compute_falloff_cohesion,
    FractalHierarchy
)

# Build hierarchy
hierarchy = FractalHierarchy.build("gold", max_depth=3)

# Get two nodes
node_a = hierarchy.get_all_nodes()[0]
node_b = hierarchy.get_all_nodes()[10]

# Calculate cohesions
natural = compute_natural_cohesion(node_a, node_b, hierarchy)
falloff = compute_falloff_cohesion(node_a, node_b, hierarchy)

print(f"Natural cohesion: {natural:.6f}")
print(f"Falloff cohesion: {falloff:.6f}")
print(f"Distance: {hierarchy.get_hierarchical_distance(node_a, node_b)}")
```

## Configuration

### Environment Variables

- **EXP13_ELEMENTS**: Default elements to test (default: "gold,nickel,copper")
- **EXP13_MAX_DEPTH**: Default max hierarchy depth (default: 5)
- **EXP13_SAMPLES**: Default interaction samples (default: 5000)

### Configuration File

Create a configuration file to customize default settings:

```toml
[EXP-13]
elements_to_test = ["gold", "nickel", "copper", "iron", "silver"]
population_size = 6
interaction_samples = 10000
```

## Results Format

### JSON Output Structure

```json
{
  "experiment": "EXP-13 v2",
  "test_type": "Fractal Gravity Without Falloff (Redesigned)",
  "start_time": "2025-01-01T12:00:00Z",
  "end_time": "2025-01-01T12:30:00Z",
  "total_duration_seconds": 1800.0,
  "elements_tested": ["gold", "nickel", "copper"],
  "element_results": {
    "gold": {
      "total_measurements": 5000,
      "natural_cohesion_flatness": 0.9995,
      "falloff_pattern_consistency": 0.9876,
      "element_fractal_density": 0.95,
      "cohesion_by_hierarchical_distance": {
        "1": {
          "natural": {
            "mean": 0.500000,
            "std": 0.000000,
            "count": 1000
          },
          "falloff": {
            "mean": 0.250000,
            "std": 0.000000,
            "count": 1000
          }
        },
        "2": {
          "natural": {
            "mean": 0.333333,
            "std": 0.000000,
            "count": 1000
          },
          "falloff": {
            "mean": 0.111111,
            "std": 0.000000,
            "count": 1000
          }
        }
      }
    }
  },
  "analysis": {
    "fractal_no_falloff_confirmed": true,
    "universal_falloff_mechanism": true,
    "mass_fractal_density_correlation": 0.85
  }
}
```

## Performance Characteristics

### Expected Results

#### Natural Cohesion
- **Flatness Score**: 0.99+ (nearly perfect constancy)
- **Variation**: < 1% across hierarchical distances
- **Pattern**: Should be identical across all elements

#### Falloff Pattern
- **Consistency Score**: 0.90+ (strong inverse-square correlation)
- **Pattern**: Identical mathematical behavior across elements
- **Exponent**: Should follow 1/distance² pattern

#### Element Properties
- **Gold**: Highest fractal density (0.95)
- **Silver**: High fractal density (0.90)
- **Copper**: Medium fractal density (0.85)
- **Nickel**: Lower fractal density (0.80)
- **Iron**: Lowest fractal density (0.75)

### Benchmark Performance

- **Small Scale** (< 1000 nodes): < 30 seconds
- **Medium Scale** (1000-10000 nodes): 2-5 minutes
- **Large Scale** (> 10000 nodes): 5-15 minutes

## Integration

### With Other Experiments

EXP-13 integrates with other experiments for comprehensive analysis:

```python
from fractalstat.exp13_fractal_gravity import FractalGravityExperiment
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment

# Run gravity test first
gravity_exp = FractalGravityExperiment(
    elements_to_test=["gold", "nickel"],
    max_hierarchy_depth=4
)
gravity_results = gravity_exp.run()

# Use gravity insights in benchmark
benchmark_exp = BenchmarkComparisonExperiment(
    sample_size=50000,
    benchmark_systems=["fractalstat", "sha256", "uuid"]
)
benchmark_results, _ = benchmark_exp.run()
```

### With Configuration System

```python
from fractalstat.config import ExperimentConfig
from fractalstat.exp13_fractal_gravity import FractalGravityExperiment

# Load configuration
config = ExperimentConfig()
elements = config.get("EXP-13", "elements_to_test", ["gold", "nickel"])
max_depth = config.get("EXP-13", "population_size", 5)
samples = config.get("EXP-13", "interaction_samples", 5000)

# Create experiment with config
experiment = FractalGravityExperiment(
    elements_to_test=elements,
    max_hierarchy_depth=max_depth,
    interaction_samples=samples
)
```

## Troubleshooting

### Common Issues

#### High Memory Usage
- **Cause**: Large hierarchy depths with many elements
- **Solution**: Reduce max_hierarchy_depth or interaction_samples
- **Example**: Use `max_hierarchy_depth=4` for quick testing

#### Long Execution Time
- **Cause**: Large interaction samples or deep hierarchies
- **Solution**: Use smaller samples or shallower trees
- **Example**: Use `interaction_samples=1000` for faster execution

#### Import Errors
- **Cause**: Missing dependencies or incorrect imports
- **Solution**: Check module structure and dependencies
- **Example**: Ensure `numpy` is available for correlation calculations

### Performance Optimization

#### Memory Optimization
```python
# Use smaller hierarchies for memory-constrained environments
experiment = FractalGravityExperiment(
    elements_to_test=["gold"],
    max_hierarchy_depth=3,  # Reduced from 5
    interaction_samples=1000  # Reduced from 5000
)
```

#### Speed Optimization
```python
# Use fewer samples for faster execution
experiment = FractalGravityExperiment(
    elements_to_test=["gold", "nickel"],
    max_hierarchy_depth=4,
    interaction_samples=1000  # Reduced from 5000
)
```

## Contributing

### Adding New Elements

To add support for new elements:

1. **Update Element Density Mapping**:
```python
def get_element_fractal_density(element: str) -> float:
    element_densities = {
        "gold": 0.95,
        "silver": 0.90,
        "copper": 0.85,
        "nickel": 0.80,
        "iron": 0.75,
        "platinum": 0.98,  # Add new element
    }
    return element_densities.get(element, 0.8)
```

2. **Test Integration**:
```python
# Test the new element
experiment = FractalGravityExperiment(
    elements_to_test=["platinum"]
)
results = experiment.run()
print(f"Platinum fractal density: {results.element_results['platinum'].element_fractal_density}")
```

### Custom Cohesion Functions

To implement custom cohesion calculations:

1. **Create Custom Function**:
```python
def custom_natural_cohesion(node_a: FractalNode, node_b: FractalNode, hierarchy: FractalHierarchy) -> float:
    distance = hierarchy.get_hierarchical_distance(node_a, node_b)
    # Custom cohesion formula
    return 1.0 / (distance ** 1.5)  # Different exponent
```

2. **Integrate with Experiment**:
```python
# Modify the experiment to use custom functions
class CustomFractalGravityExperiment(FractalGravityExperiment):
    def run_hierarchical_gravity_test_for_element(self, element_type: str) -> ElementGravityResults:
        # Use custom cohesion functions
        # ... existing logic with custom functions
        pass
```

## License

This module is part of the FractalSemantics project and is licensed under the same terms as the main project.

## Support

For questions, issues, or feature requests related to EXP-13:

1. Check the [main documentation](../../README.md)
2. Review the [configuration guide](../../docs/CONFIGURATION.md)
3. Examine [example usage](../../examples/)
4. Create an issue in the project repository

## See Also

- [EXP-12: Benchmark Comparison](../exp12_benchmark_comparison/)
- [EXP-14: Atomic Fractal Mapping](../exp14_atomic_fractal_mapping/)
- [Configuration Guide](../../docs/CONFIGURATION.md)
- [API Reference](../../docs/API_REFERENCE.md)