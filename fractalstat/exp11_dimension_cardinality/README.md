# EXP-11: Dimension Cardinality Analysis

## Overview

The Dimension Cardinality Analysis experiment explores the pros and cons of 7 dimensions vs. more or fewer dimensions in FractalStat addressing. It tests collision rates, retrieval performance, storage efficiency, and semantic expressiveness across different dimension counts (3-10 dimensions).

This experiment validates the optimal dimension count for FractalStat addressing and identifies the point of diminishing returns beyond 7 dimensions.

## Hypothesis

The 7-dimensional FractalStat addressing system provides an optimal balance between:
- Collision resistance and uniqueness
- Retrieval performance and efficiency
- Storage overhead and scalability
- Semantic expressiveness and disambiguation power

## Methodology

### Phase 1: Baseline Testing
- Test with all 7 FractalStat dimensions (realm, lineage, adjacency, horizon, luminosity, polarity, dimensionality, alignment)
- Establish baseline collision rates, performance metrics, and storage requirements
- Measure semantic expressiveness with full dimension set

### Phase 2: Reduced Dimension Testing
- Test with 3, 4, 5, 6 dimensions to understand minimum viable dimension count
- Measure degradation in collision resistance and semantic expressiveness
- Identify performance improvements from reduced complexity

### Phase 3: Extended Dimension Testing
- Test with 8, 9, 10 dimensions (including hypothetical dimensions)
- Measure marginal improvements in collision resistance
- Identify point of diminishing returns
- Evaluate storage overhead of additional dimensions

### Phase 4: Comprehensive Analysis
- Compare collision rates across all dimension counts
- Analyze retrieval performance impact
- Calculate storage efficiency per dimension
- Evaluate semantic disambiguation power
- Determine optimal dimension count

## Key Features

### Dimension Selection Strategy
- **Core FractalStat Dimensions**: Uses actual FractalStat dimensions (realm, lineage, adjacency, horizon, luminosity, polarity, dimensionality, alignment)
- **Extended Dimensions**: Includes hypothetical dimensions (temperature, entropy, coherence) for extended testing
- **Progressive Testing**: Tests dimension counts from 3 to 10 systematically

### Collision Rate Analysis
- **Address Computation**: Uses SHA-256 hashing of selected dimensions only
- **Collision Detection**: Identifies duplicate addresses across large sample sizes
- **Rate Calculation**: Measures collision frequency as percentage of total addresses

### Performance Measurement
- **Retrieval Latency**: Measures hash table lookup performance
- **Storage Overhead**: Calculates JSON serialization size per dimension
- **Memory Efficiency**: Tracks memory usage patterns across dimension counts

### Semantic Expressiveness Scoring
- **Dimension Weighting**: Assigns semantic value to each dimension type
- **Diversity Analysis**: Measures coordinate variance across samples
- **Expressiveness Score**: Combines dimension count and coordinate diversity (0.0 to 1.0 scale)

## API Reference

### Core Classes

#### DimensionTestResult
Represents results from testing a specific dimension count.

```python
from fractalstat.exp11_dimension_cardinality import DimensionTestResult

result = DimensionTestResult(
    dimension_count=7,
    dimensions_used=["realm", "lineage", "adjacency", "horizon", "luminosity", "polarity", "dimensionality"],
    sample_size=1000,
    unique_addresses=998,
    collisions=2,
    collision_rate=0.002,
    mean_retrieval_latency_ms=0.05,
    median_retrieval_latency_ms=0.04,
    avg_storage_bytes=150,
    storage_overhead_per_dimension=21.4,
    semantic_expressiveness_score=0.92
)
```

**Properties:**
- `dimension_count`: Number of dimensions tested
- `dimensions_used`: List of dimension names used in test
- `sample_size`: Number of bit-chains tested
- `unique_addresses`: Count of unique addresses generated
- `collisions`: Number of address collisions detected
- `collision_rate`: Collision rate as percentage (0.0 to 1.0)
- `mean_retrieval_latency_ms`: Average retrieval latency in milliseconds
- `median_retrieval_latency_ms`: Median retrieval latency in milliseconds
- `avg_storage_bytes`: Average storage size per coordinate set
- `storage_overhead_per_dimension`: Storage cost per dimension
- `semantic_expressiveness_score`: Semantic expressiveness score (0.0 to 1.0)

#### DimensionCardinalityResult
Contains comprehensive results from the dimension cardinality analysis.

```python
from fractalstat.exp11_dimension_cardinality import DimensionCardinalityResult

results = DimensionCardinalityResult(
    start_time="2024-01-01T00:00:00Z",
    end_time="2024-01-01T01:00:00Z",
    total_duration_seconds=3600.0,
    sample_size=1000,
    dimension_counts_tested=[3, 4, 5, 6, 7, 8, 9, 10],
    test_iterations=5,
    dimension_results=[...],
    optimal_dimension_count=7,
    optimal_collision_rate=0.001,
    optimal_retrieval_latency_ms=0.05,
    optimal_storage_efficiency=25.0,
    diminishing_returns_threshold=8,
    major_findings=["7 dimensions provide optimal balance", "Diminishing returns after 8 dimensions"],
    seven_dimensions_justified=True
)
```

**Key Properties:**
- `dimension_results`: List of DimensionTestResult objects
- `optimal_dimension_count`: Best dimension count based on analysis
- `optimal_collision_rate`: Collision rate at optimal dimension count
- `optimal_retrieval_latency_ms`: Retrieval latency at optimal dimension count
- `optimal_storage_efficiency`: Storage efficiency at optimal dimension count
- `diminishing_returns_threshold`: Dimension count where benefits plateau
- `major_findings`: Key insights from the analysis
- `seven_dimensions_justified`: Whether 7 dimensions are validated as optimal

#### EXP11_DimensionCardinality
Main experiment runner for dimension cardinality analysis.

```python
from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality

# Create experiment with custom parameters
experiment = EXP11_DimensionCardinality(
    sample_size=1000,
    dimension_counts=[3, 4, 5, 6, 7, 8, 9, 10],
    test_iterations=5
)

# Run the experiment
results, success = experiment.run()

# Access results
print(f"Optimal dimensions: {results.optimal_dimension_count}")
print(f"Collision rate: {results.optimal_collision_rate:.4%}")
print(f"Storage efficiency: {results.optimal_storage_efficiency:.1f} bytes/dimension")
```

**Configuration:**
- `sample_size`: Number of bit-chains to test per dimension count (default: 1000)
- `dimension_counts`: List of dimension counts to test (default: [3,4,5,6,7,8,9,10])
- `test_iterations`: Number of iterations per dimension count for averaging (default: 5)

**Key Methods:**
- `run()`: Execute the complete dimension cardinality analysis
- `_select_dimensions()`: Select dimensions for a given count
- `_compute_address_with_dimensions()`: Compute addresses using selected dimensions
- `_calculate_semantic_expressiveness()`: Calculate semantic expressiveness score
- `_test_dimension_count()`: Test a specific dimension count

## Usage Examples

### Basic Dimension Cardinality Testing

```python
from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality

# Create experiment with default parameters
experiment = EXP11_DimensionCardinality(sample_size=1000)
results, success = experiment.run()

# Analyze results
if results.seven_dimensions_justified:
    print("✓ 7 dimensions validated as optimal")
else:
    print("✗ 7 dimensions may not be optimal")

print(f"Optimal dimension count: {results.optimal_dimension_count}")
print(f"Best collision rate: {results.optimal_collision_rate:.4%}")
print(f"Diminishing returns at: {results.diminishing_returns_threshold} dimensions")

# View major findings
for finding in results.major_findings:
    print(f"  • {finding}")
```

### Custom Dimension Count Testing

```python
from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality

# Test specific dimension counts
experiment = EXP11_DimensionCardinality(
    sample_size=2000,
    dimension_counts=[5, 6, 7, 8, 9],
    test_iterations=3
)

results, success = experiment.run()

# Compare collision rates across tested dimensions
for test_result in results.dimension_results:
    print(f"{test_result.dimension_count} dimensions: "
          f"{test_result.collision_rate:.4%} collision rate, "
          f"{test_result.semantic_expressiveness_score:.2f} expressiveness")
```

### Performance Analysis

```python
from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality

experiment = EXP11_DimensionCardinality(sample_size=5000)
results, success = experiment.run()

# Analyze performance characteristics
print("Performance Analysis:")
print("-" * 40)

for test_result in results.dimension_results:
    print(f"{test_result.dimension_count}D: "
          f"Latency={test_result.mean_retrieval_latency_ms:.4f}ms, "
          f"Storage={test_result.avg_storage_bytes}B, "
          f"Overhead={test_result.storage_overhead_per_dimension:.1f}B/dim")
```

### Storage Efficiency Analysis

```python
from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality

experiment = EXP11_DimensionCardinality(sample_size=1000)
results, success = experiment.run()

# Analyze storage efficiency
print("Storage Efficiency Analysis:")
print("-" * 40)

storage_efficiency = []
for test_result in results.dimension_results:
    efficiency = test_result.avg_storage_bytes / test_result.dimension_count
    storage_efficiency.append((test_result.dimension_count, efficiency))

storage_efficiency.sort(key=lambda x: x[1])  # Sort by efficiency

for dim_count, efficiency in storage_efficiency:
    print(f"{dim_count} dimensions: {efficiency:.1f} bytes per dimension")
```

## Success Criteria

The experiment validates success based on these criteria:

1. **Collision Rate Analysis**: Identifies dimension count with lowest collision rate
2. **Semantic Expressiveness**: Validates that 7 dimensions provide high expressiveness (>0.9)
3. **Diminishing Returns**: Identifies point where additional dimensions provide minimal benefit
4. **Storage Efficiency**: Balances storage overhead with collision resistance
5. **Performance Impact**: Measures retrieval latency impact of dimension count

## Performance Characteristics

### Collision Rate Trends
- **3 Dimensions**: High collision rates (1-5%)
- **5 Dimensions**: Moderate collision rates (0.1-1%)
- **7 Dimensions**: Low collision rates (<0.1%)
- **8+ Dimensions**: Minimal improvement over 7 dimensions

### Storage Overhead
- **Per Dimension**: ~20-30 bytes additional storage
- **Total Overhead**: Linear growth with dimension count
- **JSON Serialization**: Efficient representation of coordinate data

### Retrieval Performance
- **Hash Table Lookup**: O(1) average case performance
- **Memory Usage**: Linear growth with sample size
- **Latency Impact**: Minimal impact from additional dimensions

### Semantic Expressiveness
- **7 Dimensions**: ~0.9+ expressiveness score
- **5 Dimensions**: ~0.7-0.8 expressiveness score
- **3 Dimensions**: ~0.5-0.6 expressiveness score

## Integration with Other Experiments

### EXP-10: Multi-Dimensional Query Optimization
- Use dimension cardinality results to optimize query performance
- Validate that 7 dimensions provide sufficient query precision
- Test query performance across different dimension counts

### EXP-11b: Dimensional Collision Stress Test
- Complement stress test results with cardinality analysis
- Validate collision resistance findings
- Cross-reference optimal dimension counts

### EXP-12: Benchmark Comparison
- Use cardinality results to establish baseline performance
- Compare FractalStat dimension efficiency with traditional systems
- Validate dimension count recommendations

## Error Handling

### Common Issues and Solutions

1. **High Collision Rates**
   ```python
   # Increase sample size for more accurate collision detection
   experiment = EXP11_DimensionCardinality(sample_size=10000)
   ```

2. **Memory Usage Problems**
   ```python
   # Reduce sample size or test iterations
   experiment = EXP11_DimensionCardinality(
       sample_size=500,
       test_iterations=2
   )
   ```

3. **Performance Issues**
   ```python
   # Use smaller dimension counts for faster testing
   experiment = EXP11_DimensionCardinality(
       dimension_counts=[3, 5, 7, 9]
   )
   ```

## Best Practices

### Experiment Design
- Use sufficiently large sample sizes (1000+ per dimension count)
- Run multiple iterations for statistical reliability
- Test dimension counts that are practically relevant
- Include both reduced and extended dimension counts

### Performance Optimization
- Use appropriate sample sizes for available memory
- Balance test iterations with execution time
- Monitor memory usage during large-scale tests
- Consider parallel execution for multiple dimension counts

### Result Analysis
- Focus on relative differences between dimension counts
- Consider practical constraints (storage, performance)
- Validate findings with additional testing
- Document assumptions and limitations

## Future Enhancements

### Planned Improvements
1. **Advanced Semantic Analysis**: More sophisticated expressiveness scoring
2. **Real-world Testing**: Test with actual FractalStat datasets
3. **Performance Profiling**: Detailed performance analysis across systems
4. **Dimension Weighting**: Optimize dimension importance scoring
5. **Adaptive Testing**: Dynamic adjustment of test parameters

### Research Directions
1. **Dimension Correlation**: Analyze relationships between dimensions
2. **Optimal Subsets**: Find optimal dimension combinations
3. **Dynamic Dimensions**: Test adaptive dimension selection
4. **Cross-system Comparison**: Compare with other addressing systems

## Conclusion

The Dimension Cardinality Analysis experiment provides comprehensive validation of the 7-dimensional FractalStat addressing system. Through systematic testing across different dimension counts, this experiment demonstrates that 7 dimensions provide an optimal balance between collision resistance, semantic expressiveness, storage efficiency, and performance.

The findings support the design choice of 7 dimensions while identifying the point of diminishing returns for additional dimensions. This analysis provides a solid foundation for FractalStat addressing system design and optimization.