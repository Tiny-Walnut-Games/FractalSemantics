# Experimental Configuration Guide

## Overview

FractalStat uses a **feature flag system** to configure all 10 validation experiments. This approach provides:

- **Reproducibility**: Lock configuration for publication
- **Flexibility**: Run experiments with different parameters without code changes
- **Environment-specific settings**: Optimize for dev, CI/CD, or production
- **A/B Testing**: Test hypothesis variations without branching code

## Configuration Files

### Location

All configuration files are located in `fractalstat/config/`:

```
fractalstat/config/
├── __init__.py
├── experiments.toml          # Default configuration (production)
├── experiments.dev.toml      # Development overrides
├── experiments.ci.toml       # CI/CD overrides
└── feature_flags.py          # Configuration loader
```

### File Precedence

Configuration is loaded with the following precedence (highest to lowest):

1. **Environment-specific config** (`experiments.{env}.toml`)
2. **Base config** (`experiments.toml`)

The environment is determined by the `FRACTALSTAT_ENV` environment variable (defaults to `dev`).

## Configuration Structure

### experiments.toml (Default)

```toml
[experiments]
# List of enabled experiments
enabled = [
    "EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05",
    "EXP-06", "EXP-07", "EXP-08", "EXP-09", "EXP-10"
]

# EXP-01: Address Uniqueness Test
[experiments.EXP-01]
name = "Address Uniqueness Test"
enabled = true
sample_size = 1000          # Bit-chains to generate per iteration
iterations = 10             # Number of test iterations
quick_mode = false

# EXP-04: Fractal Scaling
[experiments.EXP-04]
name = "Fractal Scaling"
enabled = true
quick_mode = true           # If false, test up to 1M scale
scales = [1000, 10000, 100000]
num_retrievals = 1000
timeout_seconds = 300

# EXP-10: Bob the Skeptic
[experiments.EXP-10]
name = "Bob the Skeptic"
enabled = true
duration_minutes = 30
queries_per_second_target = 10
max_concurrent_queries = 50
api_base_url = "http://localhost:8000"
bob_coherence_high = 0.85
bob_entanglement_low = 0.30
bob_consistency_threshold = 0.85
```

### experiments.dev.toml (Development)

Optimized for fast iteration:

```toml
[experiments]
# Enable only a subset for quick testing
enabled = ["EXP-01", "EXP-02", "EXP-04", "EXP-05"]

[experiments.EXP-01]
sample_size = 100           # Reduced from 1000
iterations = 3              # Reduced from 10
quick_mode = true

[experiments.EXP-04]
quick_mode = true
scales = [100, 1000]        # Smaller scales
num_retrievals = 100
```

### experiments.ci.toml (CI/CD)

Balanced for pipeline speed:

```toml
[experiments]
# Enable all experiments but with reduced parameters
enabled = [
    "EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05",
    "EXP-06", "EXP-07", "EXP-08", "EXP-09", "EXP-10"
]

[experiments.EXP-01]
sample_size = 500           # Balanced for CI
iterations = 5

[experiments.EXP-10]
duration_minutes = 2        # Quick stress test for CI
queries_per_second_target = 5
max_concurrent_queries = 20
```

## Using Different Environments

### Command Line

```bash
# Development (fast iteration)
export FRACTALSTAT_ENV=dev
python -m fractalstat.fractalstat_experiments

# CI/CD (balanced testing)
export FRACTALSTAT_ENV=ci
python -m fractalstat.exp04_fractal_scaling

# Production (full validation)
export FRACTALSTAT_ENV=production
python -m fractalstat.exp05_compression_expansion

# Or unset for default (dev)
unset FRACTALSTAT_ENV
python -m fractalstat.fractalstat_experiments
```

### GitLab CI/CD

The `.gitlab-ci.yml` file automatically sets `FRACTALSTAT_ENV=ci`:

```yaml
validate_experiments:
  variables:
    FRACTALSTAT_ENV: "ci"
  script:
    - pip install -e .
    - python -m fractalstat.fractalstat_experiments
```

## Programmatic Access

### Basic Usage

```python
from fractalstat.config import ExperimentConfig

# Load configuration
config = ExperimentConfig()

# Check if experiment is enabled
if config.is_enabled("EXP-01"):
    # Get configuration values
    sample_size = config.get("EXP-01", "sample_size", 1000)
    iterations = config.get("EXP-01", "iterations", 10)
    
    # Run experiment with config values
    exp01 = EXP01_AddressUniqueness(
        sample_size=sample_size,
        iterations=iterations
    )
    exp01.run()
```

### Advanced Usage

```python
from fractalstat.config import ExperimentConfig

config = ExperimentConfig()

# Get all configuration for an experiment
exp04_config = config.get_all("EXP-04")
print(exp04_config)
# {'name': 'Fractal Scaling', 'quick_mode': True, 'scales': [1000, 10000, 100000], ...}

# Get list of enabled experiments
enabled = config.get_enabled_experiments()
print(enabled)
# ['EXP-01', 'EXP-02', 'EXP-04', 'EXP-05']

# Get current environment
env = config.get_environment()
print(env)
# 'dev'
```

## Reproducibility for Publication

### Locking Configuration

For publication, the exact configuration used must be archived:

1. **Tag the release**:
   ```bash
   git tag -a v1.0.0 -m "Publication release"
   git push origin v1.0.0
   ```

2. **Archive includes config**:
   The `scripts/create_publication_archive.py` script automatically includes:
   - `config/experiments.toml`
   - `config/experiments.dev.toml`
   - `config/experiments.ci.toml`
   - `config/feature_flags.py`

3. **Document in paper**:
   ```
   All experiments used configuration version v1.0.0.
   See archived config/experiments.toml for exact parameters.
   ```

### Zenodo Archive

When creating a Zenodo DOI, include:
- The complete `config/` directory
- A reference to the configuration version in the metadata

### Reproducing Results

To reproduce published results:

```bash
# Clone at specific tag
git clone --branch v1.0.0 https://gitlab.com/tiny-walnut-games/fractalstat.git
cd fractalstat

# Install dependencies
pip install -e .

# Configuration is already locked at v1.0.0
# Run experiments (uses archived config)
python -m fractalstat.fractalstat_experiments
```

## Configuration Parameters by Experiment

### EXP-01: Address Uniqueness
- `sample_size` (int): Bit-chains per iteration (default: 1000)
- `iterations` (int): Number of test iterations (default: 10)
- `quick_mode` (bool): Reduce sample_size to 100 (default: false)

### EXP-02: Retrieval Efficiency
- `query_count` (int): Queries per scale (default: 1000)
- `scales` (list[int]): Test scales (default: [1000, 10000, 100000])

### EXP-03: Dimension Necessity
- `sample_size` (int): Bit-chains for ablation (default: 1000)

### EXP-04: Fractal Scaling
- `quick_mode` (bool): Skip 1M scale (default: true)
- `scales` (list[int]): Test scales (default: [1000, 10000, 100000])
- `num_retrievals` (int): Queries per scale (default: 1000)
- `timeout_seconds` (int): Test timeout (default: 300)

### EXP-05: Compression/Expansion
- `num_bitchains` (int): Bit-chains to compress (default: 100)
- `show_samples` (bool): Print detailed paths (default: true)

### EXP-06: Entanglement Detection
- `threshold` (float): Entanglement threshold (default: 0.85)
- `sample_size` (int): Bit-chain pairs to test (default: 100)

### EXP-07: LUCA Bootstrap
- `num_entities` (int): Entities to derive from LUCA (default: 10)
- `max_generations` (int): Maximum lineage depth (default: 5)

### EXP-08: RAG Integration
- `api_base_url` (str): RAG API endpoint (default: "http://localhost:8000")
- `num_queries` (int): Test queries (default: 50)
- `timeout_seconds` (int): Query timeout (default: 30)

### EXP-09: Concurrency
- `num_workers` (int): Concurrent workers (default: 4)
- `operations_per_worker` (int): Operations per worker (default: 100)
- `test_duration_seconds` (int): Max test duration (default: 60)

### EXP-10: Bob the Skeptic
- `duration_minutes` (int): Stress test duration (default: 30)
- `queries_per_second_target` (int): Target QPS (default: 10)
- `max_concurrent_queries` (int): Max concurrent (default: 50)
- `api_base_url` (str): API endpoint (default: "http://localhost:8000")
- `bob_coherence_high` (float): Coherence threshold (default: 0.85)
- `bob_entanglement_low` (float): Entanglement threshold (default: 0.30)
- `bob_consistency_threshold` (float): Consistency threshold (default: 0.85)

## Best Practices

### Development
- Use `experiments.dev.toml` for fast iteration
- Enable only the experiments you're working on
- Use smaller sample sizes and fewer iterations

### CI/CD
- Use `experiments.ci.toml` for balanced testing
- Enable all experiments to catch regressions
- Use moderate sample sizes for reasonable pipeline times

### Publication
- Use `experiments.toml` (production config)
- Run full-scale tests (disable quick_mode)
- Archive the exact configuration used
- Document configuration version in paper

### Tuning
- Start with default values
- Adjust based on your hardware and time constraints
- Document any changes in your research notes
- Consider creating custom environment configs for specific use cases

## Troubleshooting

### Config Not Loading

```python
# Check if config file exists
from pathlib import Path
config_path = Path("fractalstat/config/experiments.toml")
print(config_path.exists())  # Should be True

# Check environment variable
import os
print(os.getenv("FRACTALSTAT_ENV", "dev"))
```

### Missing tomli Dependency

For Python < 3.11, install tomli:

```bash
pip install tomli
```

### Config Syntax Errors

Validate TOML syntax:

```bash
python -c "import tomli; tomli.load(open('fractalstat/config/experiments.toml', 'rb'))"
```

## See Also

- `fractalstat/config/feature_flags.py` - Implementation details
- `README.md` - Quick start guide
- `.gitlab-ci.yml` - CI/CD configuration
- `scripts/create_publication_archive.py` - Publication archiving
