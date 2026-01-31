# EXP-12: Benchmark Comparison

## Overview

EXP-12 provides comprehensive benchmark comparison capabilities for evaluating FractalStat against established addressing and indexing systems. This experiment validates FractalStat's performance, collision rates, storage efficiency, and semantic expressiveness against industry-standard systems.

## Systems Compared

### 1. UUID/GUID System
- **Purpose**: 128-bit random identifiers
- **Strengths**: Simple, widely supported
- **Limitations**: No semantic meaning, random collisions
- **Use Cases**: Basic entity identification

### 2. SHA-256 Content Addressing
- **Purpose**: Content-based addressing (Git-style)
- **Strengths**: Content verification, deterministic
- **Limitations**: No semantic structure, fixed length
- **Use Cases**: Content-addressable storage, version control

### 3. Vector Database System
- **Purpose**: Similarity search and semantic matching
- **Strengths**: Semantic similarity, flexible queries
- **Limitations**: Approximate results, storage overhead
- **Use Cases**: AI/ML applications, recommendation systems

### 4. Graph Database System
- **Purpose**: Relationship traversal and graph queries
- **Strengths**: Excellent relationship support, pattern matching
- **Limitations**: Complex queries, specialized use cases
- **Use Cases**: Social networks, knowledge graphs

### 5. Traditional RDBMS
- **Purpose**: Structured data with indexes
- **Strengths**: Complex queries, ACID properties
- **Limitations**: Schema rigidity, scaling challenges
- **Use Cases**: Enterprise applications, transactional systems

### 6. FractalStat System
- **Purpose**: 7-dimensional semantic addressing
- **Strengths**: Multi-dimensional semantics, deterministic
- **Limitations**: Higher storage overhead, complexity
- **Use Cases**: Semantic addressing, hierarchical systems

## Key Metrics

### Uniqueness Metrics
- **Collision Rate**: Percentage of duplicate addresses
- **Unique Addresses**: Count of distinct identifiers
- **Address Distribution**: Statistical properties of generated addresses

### Retrieval Metrics
- **Mean Latency**: Average retrieval time in milliseconds
- **Median Latency**: 50th percentile retrieval time
- **P95/P99 Latency**: 95th/99th percentile retrieval times
- **Query Throughput**: Queries per second capability

### Storage Metrics
- **Average Storage**: Bytes per entity
- **Total Storage**: Overall storage requirements
- **Storage Efficiency**: Storage overhead ratio

### Semantic Capabilities
- **Expressiveness Score**: 0.0 to 1.0 semantic richness
- **Relationship Support**: 0.0 to 1.0 relationship capabilities
- **Query Flexibility**: 0.0 to 1.0 query complexity support

## API Reference

### BenchmarkComparisonExperiment

Main experiment runner for benchmark comparison.

```python
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment

# Initialize experiment
experiment = BenchmarkComparisonExperiment(
    sample_size=100000,           # Number of entities to test
    benchmark_systems=[           # Systems to compare
        "uuid", 
        "sha256", 
        "vector_db", 
        "graph_db", 
        "rdbms", 
        "fractalstat"
    ],
    scales=[10000, 100000, 1000000],  # Test scales
    num_queries=1000              # Number of retrieval queries
)

# Run benchmark
results, success = experiment.run()

# Save results
from fractalstat.exp12_benchmark_comparison import save_results
save_results(results)
```

#### Parameters

- **sample_size** (int): Number of entities to generate and test (default: 100000)
- **benchmark_systems** (List[str]): List of system names to benchmark (default: all systems)
- **scales** (List[int]): List of scales to test (default: [10000, 100000, 1000000])
- **num_queries** (int): Number of retrieval queries per scale (default: 1000)

#### Methods

- **run()** → Tuple[BenchmarkComparisonResult, bool]: Execute the benchmark comparison
- **_create_system(system_name: str)** → BenchmarkSystem: Create system instance
- **_benchmark_system(system: BenchmarkSystem, scale: int)** → SystemBenchmarkResult: Benchmark single system

### SystemBenchmarkResult

Results for a single system benchmark.

```python
from fractalstat.exp12_benchmark_comparison import SystemBenchmarkResult

# Access results
result = SystemBenchmarkResult(
    system_name="FractalStat",
    scale=100000,
    num_queries=1000,
    unique_addresses=99995,
    collisions=5,
    collision_rate=0.00005,
    mean_retrieval_latency_ms=0.1234,
    median_retrieval_latency_ms=0.1100,
    p95_retrieval_latency_ms=0.2500,
    p99_retrieval_latency_ms=0.5000,
    avg_storage_bytes_per_entity=128,
    total_storage_bytes=12800000,
    semantic_expressiveness=0.95,
    relationship_support=0.80,
    query_flexibility=0.90
)

# Convert to dictionary
result_dict = result.to_dict()
```

#### Fields

- **system_name** (str): Name of the benchmarked system
- **scale** (int): Number of entities tested
- **num_queries** (int): Number of retrieval queries executed
- **unique_addresses** (int): Count of unique addresses generated
- **collisions** (int): Number of address collisions
- **collision_rate** (float): Collision rate as percentage
- **mean_retrieval_latency_ms** (float): Average retrieval latency
- **median_retrieval_latency_ms** (float): Median retrieval latency
- **p95_retrieval_latency_ms** (float): 95th percentile retrieval latency
- **p99_retrieval_latency_ms** (float): 99th percentile retrieval latency
- **avg_storage_bytes_per_entity** (int): Average storage per entity
- **total_storage_bytes** (int): Total storage used
- **semantic_expressiveness** (float): Semantic expressiveness score (0.0-1.0)
- **relationship_support** (float): Relationship support score (0.0-1.0)
- **query_flexibility** (float): Query flexibility score (0.0-1.0)

### BenchmarkComparisonResult

Complete results from benchmark comparison.

```python
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonResult

# Access comparative results
comparison = BenchmarkComparisonResult(
    start_time="2025-01-01T12:00:00Z",
    end_time="2025-01-01T12:30:00Z",
    total_duration_seconds=1800.0,
    sample_size=100000,
    scales_tested=[100000],
    num_queries=1000,
    systems_tested=["uuid", "sha256", "fractalstat"],
    system_results=[...],  # List of SystemBenchmarkResult
    best_collision_rate_system="fractalstat",
    best_retrieval_latency_system="uuid",
    best_storage_efficiency_system="sha256",
    best_semantic_expressiveness_system="fractalstat",
    best_overall_system="fractalstat",
    fractalstat_rank_collision=1,
    fractalstat_rank_retrieval=3,
    fractalstat_rank_storage=6,
    fractalstat_rank_semantic=1,
    fractalstat_overall_score=0.85,
    major_findings=["FractalStat excels at semantic expressiveness"],
    fractalstat_competitive=True
)

# Convert to dictionary
comparison_dict = comparison.to_dict()
```

#### Fields

- **start_time** (str): Experiment start timestamp (ISO format)
- **end_time** (str): Experiment end timestamp (ISO format)
- **total_duration_seconds** (float): Total execution time
- **sample_size** (int): Number of entities tested
- **scales_tested** (List[int]): Scales that were actually tested
- **num_queries** (int): Number of queries per scale
- **systems_tested** (List[str]): Systems that were successfully tested
- **system_results** (List[SystemBenchmarkResult]): Individual system results
- **best_collision_rate_system** (str): System with lowest collision rate
- **best_retrieval_latency_system** (str): System with best retrieval latency
- **best_storage_efficiency_system** (str): System with best storage efficiency
- **best_semantic_expressiveness_system** (str): System with best semantic expressiveness
- **best_overall_system** (str): System with best overall score
- **fractalstat_rank_collision** (int): FractalStat collision rate ranking
- **fractalstat_rank_retrieval** (int): FractalStat retrieval latency ranking
- **fractalstat_rank_storage** (int): FractalStat storage efficiency ranking
- **fractalstat_rank_semantic** (int): FractalStat semantic expressiveness ranking
- **fractalstat_overall_score** (float): FractalStat overall score (0.0-1.0)
- **major_findings** (List[str]): Key insights from the comparison
- **fractalstat_competitive** (bool): Whether FractalStat is competitive overall

## Usage Examples

### Basic Benchmark

```python
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment

# Quick benchmark with smaller dataset
experiment = BenchmarkComparisonExperiment(
    sample_size=10000,
    benchmark_systems=["uuid", "sha256", "fractalstat"],
    scales=[10000],
    num_queries=100
)

results, success = experiment.run()
print(f"FractalStat competitive: {results.fractalstat_competitive}")
print(f"FractalStat overall score: {results.fractalstat_overall_score}")
```

### Custom System Configuration

```python
from fractalstat.exp12_benchmark_comparison import (
    BenchmarkComparisonExperiment,
    SystemBenchmarkResult
)

# Custom benchmark with specific systems
experiment = BenchmarkComparisonExperiment(
    sample_size=50000,
    benchmark_systems=["fractalstat", "vector_db", "rdbms"],
    scales=[50000],
    num_queries=500
)

results, success = experiment.run()

# Analyze FractalStat performance
fractalstat_result = next(
    r for r in results.system_results 
    if r.system_name == "FractalStat"
)

print(f"FractalStat collision rate: {fractalstat_result.collision_rate:.6%}")
print(f"FractalStat retrieval latency: {fractalstat_result.mean_retrieval_latency_ms:.4f}ms")
print(f"FractalStat semantic expressiveness: {fractalstat_result.semantic_expressiveness:.2f}")
```

### Performance Analysis

```python
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment

# Large-scale performance test
experiment = BenchmarkComparisonExperiment(
    sample_size=1000000,
    benchmark_systems=["fractalstat", "sha256"],
    scales=[100000, 1000000],
    num_queries=5000
)

results, success = experiment.run()

# Compare collision rates
for result in results.system_results:
    print(f"{result.system_name}: {result.collision_rate:.8%} collision rate")
    
# Compare storage efficiency
for result in results.system_results:
    print(f"{result.system_name}: {result.avg_storage_bytes_per_entity} bytes/entity")
```

## Configuration

### Environment Variables

- **EXP12_SAMPLE_SIZE**: Default sample size (default: 100000)
- **EXP12_NUM_QUERIES**: Default number of queries (default: 1000)
- **EXP12_SCALES**: Default test scales (default: [10000, 100000, 1000000])

### Configuration File

Create a configuration file to customize default settings:

```toml
[EXP-12]
sample_size = 50000
benchmark_systems = ["uuid", "sha256", "fractalstat"]
scales = [10000, 50000]
num_queries = 500
```

## Results Format

### JSON Output Structure

```json
{
  "experiment": "EXP-12",
  "test_type": "Benchmark Comparison",
  "start_time": "2025-01-01T12:00:00Z",
  "end_time": "2025-01-01T12:30:00Z",
  "total_duration_seconds": 1800.0,
  "sample_size": 100000,
  "scales_tested": [100000],
  "num_queries": 1000,
  "systems_tested": ["uuid", "sha256", "fractalstat"],
  "comparative_analysis": {
    "best_collision_rate": "fractalstat",
    "best_retrieval_latency": "uuid",
    "best_storage_efficiency": "sha256",
    "best_semantic_expressiveness": "fractalstat",
    "best_overall": "fractalstat"
  },
  "fractalstat_positioning": {
    "rank_collision": 1,
    "rank_retrieval": 3,
    "rank_storage": 6,
    "rank_semantic": 1,
    "overall_score": 0.85,
    "competitive": true
  },
  "major_findings": [
    "FractalStat excels at semantic expressiveness (multi-dimensional addressing)",
    "FractalStat competitive on collision rates (deterministic addressing)"
  ],
  "system_results": [
    {
      "system_name": "FractalStat",
      "scale": 100000,
      "num_queries": 1000,
      "unique_addresses": 99995,
      "collisions": 5,
      "collision_rate": 0.00005,
      "mean_retrieval_latency_ms": 0.1234,
      "median_retrieval_latency_ms": 0.1100,
      "p95_retrieval_latency_ms": 0.2500,
      "p99_retrieval_latency_ms": 0.5000,
      "avg_storage_bytes_per_entity": 128,
      "total_storage_bytes": 12800000,
      "semantic_expressiveness": 0.95,
      "relationship_support": 0.80,
      "query_flexibility": 0.90
    }
  ]
}
```

## Performance Characteristics

### Expected Results

#### FractalStat Strengths
- **Semantic Expressiveness**: 0.90-0.95 (excellent)
- **Collision Rate**: < 0.0001% (excellent)
- **Query Flexibility**: 0.85-0.95 (excellent)

#### FractalStat Trade-offs
- **Storage Overhead**: Higher than UUID/SHA256 (7 dimensions)
- **Retrieval Latency**: Comparable to other hash-based systems
- **Complexity**: Higher than simple addressing systems

### Benchmark Performance

- **Small Scale** (< 10K entities): < 30 seconds
- **Medium Scale** (10K-100K entities): 2-5 minutes
- **Large Scale** (> 100K entities): 5-15 minutes

## Integration

### With Other Experiments

EXP-12 integrates with other experiments for comprehensive analysis:

```python
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment
from fractalstat.exp11_dimension_cardinality import EXP11_DimensionCardinality

# Run dimension cardinality analysis
dimension_exp = EXP11_DimensionCardinality(sample_size=10000)
dimension_results, _ = dimension_exp.run()

# Use optimal dimensions in benchmark
optimal_dims = dimension_results.optimal_dimension_count

# Run benchmark with optimal configuration
benchmark_exp = BenchmarkComparisonExperiment(
    sample_size=50000,
    benchmark_systems=["fractalstat", "sha256", "uuid"]
)
benchmark_results, _ = benchmark_exp.run()
```

### With Configuration System

```python
from fractalstat.config import ExperimentConfig
from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment

# Load configuration
config = ExperimentConfig()
sample_size = config.get("EXP-12", "sample_size", 100000)
benchmark_systems = config.get("EXP-12", "benchmark_systems", ["fractalstat"])

# Create experiment with config
experiment = BenchmarkComparisonExperiment(
    sample_size=sample_size,
    benchmark_systems=benchmark_systems
)
```

## Troubleshooting

### Common Issues

#### High Memory Usage
- **Cause**: Large sample sizes with many systems
- **Solution**: Reduce sample_size or benchmark_systems list
- **Example**: Use `sample_size=10000` for quick testing

#### Long Execution Time
- **Cause**: Large scales or high query counts
- **Solution**: Use smaller scales or fewer queries
- **Example**: Use `scales=[10000]` and `num_queries=100`

#### Import Errors
- **Cause**: Missing dependencies or incorrect imports
- **Solution**: Check module structure and dependencies
- **Example**: Ensure `fractalstat_entity` is available

### Performance Optimization

#### Memory Optimization
```python
# Use smaller sample sizes for memory-constrained environments
experiment = BenchmarkComparisonExperiment(
    sample_size=5000,  # Reduced from 100000
    benchmark_systems=["fractalstat", "sha256"],  # Fewer systems
    num_queries=50  # Reduced queries
)
```

#### Speed Optimization
```python
# Use smaller scales for faster execution
experiment = BenchmarkComparisonExperiment(
    sample_size=10000,
    scales=[10000],  # Single scale
    num_queries=100  # Fewer queries
)
```

## Contributing

### Adding New Systems

To add a new benchmark system:

1. **Create System Class**:
```python
class NewSystem(BenchmarkSystem):
    def __init__(self):
        super().__init__("NewSystem")
    
    def generate_address(self, entity: Any) -> str:
        # Implementation
        pass
    
    def get_semantic_expressiveness(self) -> float:
        return 0.5  # Score 0.0-1.0
    
    def get_relationship_support(self) -> float:
        return 0.3  # Score 0.0-1.0
    
    def get_query_flexibility(self) -> float:
        return 0.7  # Score 0.0-1.0
```

2. **Register System**:
```python
# In experiment.py, add to systems dictionary
systems = {
    "uuid": UUIDSystem,
    "sha256": SHA256System,
    # ... existing systems
    "new_system": NewSystem,  # Add new system
}
```

3. **Test Integration**:
```python
# Test the new system
experiment = BenchmarkComparisonExperiment(
    benchmark_systems=["new_system"]
)
results, success = experiment.run()
```

### Custom Metrics

To add custom metrics:

1. **Extend SystemBenchmarkResult**:
```python
@dataclass
class ExtendedSystemBenchmarkResult(SystemBenchmarkResult):
    custom_metric: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["custom_metric"] = self.custom_metric
        return result
```

2. **Update Benchmarking Logic**:
```python
def _benchmark_system(self, system: BenchmarkSystem, scale: int) -> ExtendedSystemBenchmarkResult:
    # ... existing logic
    custom_metric = self._calculate_custom_metric(system, scale)
    
    return ExtendedSystemBenchmarkResult(
        # ... existing fields
        custom_metric=custom_metric
    )
```

## License

This module is part of the FractalSemantics project and is licensed under the same terms as the main project.

## Support

For questions, issues, or feature requests related to EXP-12:

1. Check the [main documentation](../../README.md)
2. Review the [configuration guide](../../docs/CONFIGURATION.md)
3. Examine [example usage](../../examples/)
4. Create an issue in the project repository

## See Also

- [EXP-11: Dimension Cardinality Analysis](../exp11_dimension_cardinality/)
- [EXP-13: Fractal Gravity](../exp13_fractal_gravity/)
- [Configuration Guide](../../docs/CONFIGURATION.md)
- [API Reference](../../docs/API_REFERENCE.md)