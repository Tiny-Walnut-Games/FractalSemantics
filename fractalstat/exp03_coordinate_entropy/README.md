# EXP-03: Coordinate Space Entropy Test

This module demonstrates the successful refactoring of a large, complex experiment file (1,075 lines) into a modular, maintainable structure. The original `exp03_coordinate_entropy.py` has been broken down into focused, single-responsibility modules.

## File Organization

### Before (Monolithic)
- `exp03_coordinate_entropy.py` (1,075 lines)
  - All functionality in one massive file
  - Difficult to search and maintain
  - Mixed concerns and responsibilities

### After (Modular)
```
fractalstat/exp03_coordinate_entropy/
├── __init__.py              # Module exports and usage examples
├── entities.py              # BitChain generation and coordinate definitions (200 lines)
├── entropy_analysis.py      # Shannon entropy computation and analysis (400 lines)
├── visualization.py         # Results saving and chart generation (100 lines)
├── experiment.py            # Main experiment orchestration (50 lines)
└── README.md               # This documentation
```

**Total: ~750 lines across 5 focused modules**

## Benefits of Refactoring

### 1. **Searchability**
- Each module has a clear, specific purpose
- Functions are focused and well-documented
- Easy to find specific functionality

### 2. **Maintainability**
- Single responsibility principle applied
- Changes to one aspect don't affect others
- Clear dependency relationships

### 3. **Testability**
- Each module can be tested independently
- Mock objects are easier to create
- Unit tests are more focused

### 4. **Reusability**
- Components can be reused across experiments
- Clear interfaces between modules
- Better separation of concerns

## Usage Examples

### Basic Usage
```python
from fractalstat.exp03_coordinate_entropy import EXP03_CoordinateEntropy

# Run the complete experiment
experiment = EXP03_CoordinateEntropy(sample_size=100000)
results, success = experiment.run()
summary = experiment.get_summary()
print(f"All dimensions critical: {summary['all_critical']}")
```

### Advanced Usage
```python
from fractalstat.exp03_coordinate_entropy import (
    generate_random_bitchain, 
    compute_shannon_entropy,
    extract_coordinates
)

# Generate test data
bitchains = [generate_random_bitchain(seed=i) for i in range(1000)]

# Compute entropy for specific dimensions
coords = extract_coordinates(bitchains, ["realm", "lineage", "horizon"])
entropy = compute_shannon_entropy(coords)
print(f"Entropy: {entropy:.4f} bits")
```

## Module Breakdown

### `entities.py`
- **Purpose**: BitChain generation and coordinate entity definitions
- **Key Classes**: `EXP03_Result`, `BitChain`, `Coordinates`, `FractalStatCoordinates`
- **Key Functions**: `generate_random_bitchain()`, coordinate mapping utilities

### `entropy_analysis.py`
- **Purpose**: Core entropy computation and analysis algorithms
- **Key Classes**: `EXP03_CoordinateEntropy`
- **Key Functions**: `compute_shannon_entropy()`, `compute_expressiveness_contribution()`, `extract_coordinates()`

### `visualization.py`
- **Purpose**: Results saving and visualization generation
- **Key Functions**: `save_results()`, `plot_entropy_contributions()`

### `experiment.py`
- **Purpose**: Main experiment orchestration and entry point
- **Key Functions**: `main()`, configuration loading

## Scientific Purpose

This experiment quantifies the information-theoretic entropy contribution of each FractalStat dimension to the coordinate space, measuring how well each dimension contributes to semantic disambiguation.

**Key Concepts:**
1. **Shannon Entropy**: Measures information content of coordinate space
2. **Coordinate-Level Measurement**: Entropy measured BEFORE hashing
3. **Ablation Testing**: Remove each dimension and measure entropy loss
4. **Semantic Disambiguation**: How well dimensions separate entities

**The 8 Dimensions Analyzed:**
- realm: Domain classification (data, narrative, system, faculty, event, pattern, void)
- lineage: Generation from LUCA (temporal context)
- adjacency: Relational neighbors (graph structure)
- horizon: Lifecycle stage (genesis, emergence, peak, decay, crystallization)
- luminosity: Activity level (0-100)
- polarity: Resonance/affinity type (6 companion + 6 badge + neutral)
- dimensionality: Fractal depth (0+)
- alignment: Social/coordination dynamic type (lawful_good, chaotic_evil, etc.)

## Search Enhancement Pattern

Each module includes clear documentation and type hints:

```python
def compute_shannon_entropy(self, coordinates: List[str]) -> float:
    """
    Compute Shannon entropy of a list of coordinate representations.

    Shannon entropy H(X) = -Σ p(x) * log2(p(x))
    where p(x) is the probability of observing coordinate value x.

    Higher entropy indicates more information content and better
    discrimination between different entities.

    Args:
        coordinates: List of coordinate string representations

    Returns:
        Shannon entropy in bits
    """
```

## Documentation Standards

- Every function >10 lines has a docstring
- Complex algorithms include inline comments
- Module-level documentation explains purpose and usage
- Type hints for better IDE support

## Testing Strategy

Each module can be tested independently:

```python
# Test entropy computation
from fractalstat.exp03_coordinate_entropy.entropy_analysis import EXP03_CoordinateEntropy

def test_entropy_computation():
    experiment = EXP03_CoordinateEntropy(sample_size=1000)
    coords = ["coord1", "coord2", "coord1", "coord3"]
    entropy = experiment.compute_shannon_entropy(coords)
    assert entropy > 0
```

## Migration Guide

To migrate other large files in your project:

1. **Identify logical boundaries** - Look for natural groupings of functionality
2. **Create focused modules** - Each module should have a single, clear purpose
3. **Define clear interfaces** - Use imports and exports to manage dependencies
4. **Add comprehensive documentation** - Document each module's purpose and usage
5. **Update imports** - Ensure all existing code continues to work
6. **Test thoroughly** - Verify functionality remains unchanged

## Performance Considerations

- Modular structure has minimal performance impact
- Import overhead is negligible compared to computational work
- Better memory management through focused modules
- Easier to optimize specific components

## Future Enhancements

- Add more entropy computation methods (Rényi entropy, etc.)
- Implement additional visualization types
- Add performance benchmarks
- Create interactive analysis tools
- Integrate with other fractal experiments

This refactoring demonstrates how large, complex files can be transformed into maintainable, searchable, and extensible code while preserving all original functionality.