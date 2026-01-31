# EXP-14: Atomic Fractal Mapping

## Overview

EXP-14 tests whether electron shell structure naturally maps to fractal hierarchy, validating that atomic structure emerges from fractal principles.

**CORRECTED DESIGN**: Uses actual electron shell configuration as input, not naive Z-based mapping.

This experiment represents Phase 2 of fractal gravity validation, focusing on the fundamental question: **Does atomic structure naturally emerge from fractal hierarchy?**

## Success Criteria

- **Fractal depth matches electron shell count** (100% accuracy)
- **Branching factor correlates with valence electrons** (>0.95 correlation)
- **Node count scales as branching^depth** (exponential validation)
- **Prediction errors decrease with shell depth** (negative correlation)

## Core Components

### ElectronConfiguration
Represents the electron shell configuration for elements:

```python
@dataclass
class ElectronConfiguration:
    element: str              # Element name (e.g., "hydrogen")
    symbol: str              # Chemical symbol (e.g., "H")
    atomic_number: int       # Protons (Z)
    neutron_number: int      # Neutrons (N)
    atomic_mass: float       # Atomic mass in u
    
    # Electron shell structure (THE KEY INPUT)
    electron_config: str     # Full configuration (e.g., "1s² 2s² 2p⁶")
    shell_count: int         # Number of electron shells
    valence_electrons: int   # Valence electrons
```

### ShellBasedFractalMapping
Maps electron shells to fractal parameters:

```python
@dataclass
class ShellBasedFractalMapping:
    element: str
    electron_config: ElectronConfiguration
    
    # Direct mappings (no formulas - just count)
    fractal_depth: int        # = shell_count
    branching_factor: int     # = valence_electrons (+ nuclear adjustments)
    total_nodes: int          # = branching_factor ^ fractal_depth
    
    # Validation against EXP-13 observed densities
    predicted_density: float
    actual_density: float
    density_error: float
    
    # Structure validation (the real test)
    depth_matches_shells: bool     # fractal_depth == shell_count
    branching_matches_valence: bool # branching_factor ≈ valence_electrons
    node_growth_exponential: bool   # nodes = branching^depth
```

### AtomicFractalMappingExperiment
Main experiment runner:

```python
experiment = AtomicFractalMappingExperiment(
    elements_to_test=["hydrogen", "carbon", "gold", "uranium"]
)
results = experiment.run()
```

## Key Features

### Comprehensive Element Coverage
- **Period 1-7 elements**: Full periodic table coverage
- **Shell configurations**: Actual electron shell data
- **Nuclear complexity**: Neutron number adjustments
- **Valence analysis**: Precise valence electron counting

### Structure Validation
- **Depth accuracy**: Measures how well fractal depth matches shell count
- **Branching correlation**: Validates branching factor against valence electrons
- **Exponential growth**: Confirms node count follows branching^depth pattern
- **Error analysis**: Tracks prediction accuracy across elements

### Nuclear Complexity Adjustments
- **Neutron-rich nuclei**: Higher branching for neutron-rich elements
- **Nuclear stability**: Adjustments based on neutron-to-proton ratios
- **Element classification**: Different adjustment patterns for different element groups

## Usage Examples

### Basic Usage
```python
from fractalstat.exp14_atomic_fractal_mapping import AtomicFractalMappingExperiment

# Test specific elements
experiment = AtomicFractalMappingExperiment(
    elements_to_test=["hydrogen", "carbon", "oxygen", "gold"]
)
results = experiment.run()

# Check structure validation
print(f"Depth accuracy: {results['structure_validation']['depth_accuracy']:.1%}")
print(f"Branching accuracy: {results['structure_validation']['branching_accuracy']:.1%}")
```

### Full Periodic Table Analysis
```python
# Test all available elements
experiment = AtomicFractalMappingExperiment()  # Uses all elements
results = experiment.run()

# Analyze periodic trends
for element, mapping in results['mappings'].items():
    print(f"{element}: depth={mapping['fractal_depth']}, "
          f"branching={mapping['branching_factor']}, "
          f"nodes={mapping['total_nodes']}")
```

### Custom Element Selection
```python
# Test specific element groups
alkali_metals = ["lithium", "sodium", "potassium", "rubidium", "cesium"]
transition_metals = ["iron", "copper", "silver", "gold", "platinum"]

experiment = AtomicFractalMappingExperiment(elements_to_test=alkali_metals)
results = experiment.run()
```

## Data Structure

### Input Data
The experiment uses comprehensive electron shell data including:
- **Element properties**: Atomic number, symbol, mass, neutron count
- **Shell configurations**: Complete electron configurations
- **Shell counts**: Number of electron shells per element
- **Valence electrons**: Number of valence electrons per element

### Output Results
Complete experiment results include:
- **Mapping data**: Element-by-element fractal mappings
- **Structure validation**: Accuracy metrics for depth and branching
- **Density statistics**: Prediction accuracy and correlations
- **Success criteria**: Pass/fail status based on validation thresholds

## Validation Metrics

### Structure Validation
- **Depth Accuracy**: Percentage of elements where fractal depth equals shell count
- **Branching Accuracy**: Percentage of elements where branching factor matches valence electrons
- **Exponential Consistency**: Confirmation that node growth follows exponential pattern

### Density Prediction
- **Mean Error**: Average difference between predicted and actual densities
- **Correlation**: Linear correlation between predicted and actual densities
- **Error Distribution**: Analysis of prediction errors across different element types

## Success Criteria Implementation

The experiment validates success through:

1. **Depth Matching**: 95% of elements must have fractal depth equal to shell count
2. **Branching Correlation**: 80% of elements must have reasonable branching-valence match
3. **Exponential Growth**: 100% of elements must show exponential node growth
4. **Overall Success**: All three criteria must be met for experiment success

## Integration with Other Experiments

### EXP-13 Fractal Gravity
- Uses `get_element_fractal_density()` for actual density comparisons
- Validates that shell-based predictions correlate with gravity-based densities
- Provides cross-validation between different fractal approaches

### EXP-15 Topological Conservation
- Shares electron configuration data structures
- Validates that topological properties emerge from shell structure
- Provides foundation for conservation law testing

## Performance Characteristics

### Computational Complexity
- **Time Complexity**: O(n) where n is the number of elements tested
- **Space Complexity**: O(n) for storing mapping results
- **Memory Usage**: Minimal - primarily stores configuration data and results

### Scalability
- **Element Count**: Scales linearly with number of elements
- **Configuration Complexity**: Independent of electron configuration complexity
- **Validation Overhead**: Constant time per element for validation checks

## Error Handling

### Data Validation
- **Missing Elements**: Validates that all requested elements exist in database
- **Configuration Errors**: Handles malformed electron configurations
- **Type Safety**: Ensures all data types are correct and consistent

### Runtime Errors
- **Overflow Protection**: Caps node counts to prevent integer overflow
- **Division by Zero**: Handles edge cases in correlation calculations
- **Missing Dependencies**: Graceful handling of missing EXP-13 data

## Future Enhancements

### Potential Improvements
- **Relativistic Effects**: Include relativistic corrections for heavy elements
- **Isotope Variations**: Support for different isotopes of the same element
- **Ion States**: Handle ionized atoms with different electron counts
- **Molecular Structures**: Extend to molecular electron configurations

### Research Directions
- **Quantum Corrections**: Include quantum mechanical effects in fractal mapping
- **Periodic Trends**: Analyze fractal parameter trends across the periodic table
- **Chemical Properties**: Correlate fractal parameters with chemical behavior
- **Material Properties**: Extend to bulk material properties

## Testing and Validation

### Unit Tests
- **Entity Validation**: Test all data structure components
- **Mapping Logic**: Validate shell-to-fractal conversion algorithms
- **Statistics**: Test correlation and accuracy calculations
- **Error Handling**: Verify proper error handling and edge cases

### Integration Tests
- **EXP-13 Integration**: Test integration with fractal gravity data
- **Configuration Loading**: Validate electron configuration data loading
- **Result Processing**: Test complete result generation and formatting

### Performance Tests
- **Large Element Sets**: Test performance with full periodic table
- **Memory Usage**: Monitor memory consumption during large-scale runs
- **Computation Time**: Measure execution time for different element counts

## Conclusion

EXP-14 provides a comprehensive framework for testing whether atomic structure naturally emerges from fractal hierarchy. By using actual electron shell configurations as input, it validates that the fundamental building blocks of matter follow fractal principles.

The experiment's success would provide strong evidence that fractal geometry underlies atomic structure, supporting the broader hypothesis that fractal principles govern the organization of matter at all scales.