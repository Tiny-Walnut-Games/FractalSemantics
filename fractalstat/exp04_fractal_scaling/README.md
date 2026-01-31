# EXP-04: FractalStat Fractal Scaling Test

## Overview

The Fractal Scaling Test validates that FractalStat addressing maintains consistency and zero collisions when scaled from 1K → 10K → 100K → 1M data points. This experiment verifies the "fractal" property: self-similar behavior at all scales.

## Purpose

This experiment tests whether the FractalStat addressing system exhibits true fractal properties by:

1. **Scalability Testing**: Verifying performance and collision rates across multiple orders of magnitude
2. **Fractal Property Validation**: Ensuring self-similar behavior regardless of scale
3. **Performance Degradation Analysis**: Checking that retrieval times scale logarithmically rather than linearly

## Key Concepts

### Fractal Properties
- **Self-similarity**: System behavior should be consistent across different scales
- **Zero collisions**: Collision rate should remain zero regardless of data volume
- **Logarithmic scaling**: Retrieval performance should degrade logarithmically, not linearly

### Test Scales
- **1K**: 1,000 bit-chains (baseline)
- **10K**: 10,000 bit-chains (10x scale)
- **100K**: 100,000 bit-chains (100x scale)
- **1M**: 1,000,000 bit-chains (1000x scale, optional)

## Module Structure

```
fractalstat/exp04_fractal_scaling/
├── __init__.py          # Module exports and version info
├── __main__.py          # CLI entry point
├── entities.py          # Data models and configuration classes
├── experiment.py        # Core experiment logic
├── results.py           # Results processing and file I/O
└── README.md           # This documentation
```

## Data Models

### ScaleTestConfig
Configuration for testing a single scale level:

```python
@dataclass
class ScaleTestConfig:
    scale: int              # Number of bit-chains (1K, 10K, 100K, 1M)
    num_retrievals: int     # Number of random retrieval queries
    timeout_seconds: int    # Kill test if it takes too long
```

### ScaleTestResults
Results from testing a single scale level:

```python
@dataclass
class ScaleTestResults:
    scale: int
    num_bitchains: int
    num_addresses: int
    unique_addresses: int
    collision_count: int
    collision_rate: float
    
    # Retrieval performance metrics
    num_retrievals: int
    retrieval_times_ms: List[float]
    retrieval_mean_ms: float
    retrieval_median_ms: float
    retrieval_p95_ms: float
    retrieval_p99_ms: float
    
    # System metrics
    total_time_seconds: float
    addresses_per_second: float
```

### FractalScalingResults
Complete results from the fractal scaling test:

```python
@dataclass
class FractalScalingResults:
    start_time: str
    end_time: str
    total_duration_seconds: float
    scale_results: List[ScaleTestResults]
    
    # Degradation analysis
    collision_degradation: Optional[str]
    retrieval_degradation: Optional[str]
    is_fractal: bool
```

## Core Functions

### run_scale_test(config: ScaleTestConfig) -> ScaleTestResults
Executes EXP-01 (uniqueness) + EXP-02 (retrieval) at a single scale:

1. **Generate bit-chains**: Creates the specified number of random bit-chains
2. **Compute addresses**: Calculates FractalStat addresses and checks for collisions
3. **Build retrieval index**: Creates hash map for fast address lookup
4. **Test retrieval performance**: Measures lookup times for random queries
5. **Calculate statistics**: Computes mean, median, P95, P99 retrieval times

### analyze_degradation(results: List[ScaleTestResults]) -> Tuple[str, str, bool]
Analyzes whether the system maintains fractal properties:

- **Collision Analysis**: Verifies zero collisions at all scales
- **Retrieval Analysis**: Checks if retrieval time scales logarithmically
- **Fractal Determination**: Returns whether system exhibits fractal behavior

### run_fractal_scaling_test(quick_mode: bool = True) -> FractalScalingResults
Main orchestration function that:

1. Loads experiment configuration
2. Runs tests at multiple scales
3. Analyzes degradation patterns
4. Returns complete results with fractal analysis

## Usage

### As a Module
```python
from fractalstat.exp04_fractal_scaling import run_fractal_scaling_test, save_results

# Run with default quick mode (1K, 10K, 100K)
results = run_fractal_scaling_test(quick_mode=True)

# Save results to JSON
output_file = save_results(results)
```

### As a Script
```bash
# Quick mode (1K, 10K, 100K)
python -m fractalstat.exp04_fractal_scaling

# Full mode (1K, 10K, 100K, 1M)
python -m fractalstat.exp04_fractal_scaling --full

# Direct execution
python fractalstat/exp04_fractal_scaling/__main__.py
```

### Configuration
The experiment can be configured via the experiment configuration system:

```python
from fractalstat.config import ExperimentConfig

config = ExperimentConfig()
quick_mode = config.get("EXP-04", "quick_mode", True)
scales = config.get("EXP-04", "scales", [1_000, 10_000, 100_000])
num_retrievals = config.get("EXP-04", "num_retrievals", 1000)
```

## Success Criteria

### Collision Requirements
- **Zero collisions** at all tested scales
- **Collision rate** must remain 0.0% regardless of data volume

### Performance Requirements
- **Sub-millisecond retrieval** (mean < 2.0ms)
- **Logarithmic scaling** of retrieval times
- **Addresses per second** throughput should remain high

### Fractal Requirements
- **Self-similar behavior** across all scales
- **No linear degradation** in performance
- **Consistent collision patterns** (zero collisions)

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "experiment": "EXP-04",
  "test_type": "Fractal Scaling",
  "start_time": "2025-01-01T12:00:00Z",
  "end_time": "2025-01-01T12:05:00Z",
  "total_duration_seconds": 300.0,
  "scales_tested": 3,
  "scale_results": [
    {
      "scale": 1000,
      "num_bitchains": 1000,
      "num_addresses": 1000,
      "unique_addresses": 1000,
      "collision_count": 0,
      "collision_rate_percent": 0.0,
      "retrieval": {
        "num_queries": 1000,
        "mean_ms": 0.000123,
        "median_ms": 0.000110,
        "p95_ms": 0.000250,
        "p99_ms": 0.000500
      },
      "performance": {
        "total_time_seconds": 0.5,
        "addresses_per_second": 2000
      },
      "valid": true
    }
  ],
  "degradation_analysis": {
    "collision_degradation": "OK Zero collisions at all scales",
    "retrieval_degradation": "OK Retrieval latency scales logarithmically (1.5x for 100x scale)"
  },
  "is_fractal": true,
  "all_valid": true
}
```

## Integration with Other Experiments

This experiment builds upon:

- **EXP-01**: Uses the same collision detection logic
- **EXP-02**: Uses the same retrieval performance testing
- **EXP-03**: Complements entropy analysis with scalability testing

## Troubleshooting

### Common Issues

1. **High collision rates**: Indicates hash function issues or insufficient entropy
2. **Linear performance degradation**: Suggests algorithmic complexity problems
3. **Timeout errors**: May indicate memory or CPU constraints at large scales

### Performance Optimization

- Use `quick_mode=True` for faster testing during development
- Monitor memory usage at larger scales
- Consider increasing timeout values for very large datasets

## Dependencies

- **fractalstat.fractalstat_experiments**: For BitChain generation and address computation
- **fractalstat.config**: For experiment configuration (optional)
- **Standard library**: json, time, datetime, pathlib, typing, dataclasses, collections, statistics

## Version History

- **1.0.0**: Initial modular implementation
  - Separated entities, experiment logic, and results processing
  - Added comprehensive documentation
  - Implemented proper error handling and validation