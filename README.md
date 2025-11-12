# FractalStat - STAT7 Validation Experiments

**A complete validation suite for the STAT7 7-dimensional addressing system**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## What is FractalStat?

FractalStat is a research package containing **12 validation experiments** that prove the STAT7 addressing system works at scale. STAT7 is a 7-dimensional coordinate system for uniquely addressing data in fractal information spaces.

**The 7 Dimensions:**
- **Realm** - Domain classification (data, narrative, system, etc.)
- **Lineage** - Generation from LUCA (Last Universal Common Ancestor)
- **Adjacency** - Relational neighbors (graph connections)
- **Horizon** - Lifecycle stage (genesis, emergence, peak, decay, crystallization)
- **Resonance** - Charge/alignment (-1.0 to 1.0)
- **Velocity** - Rate of change
- **Density** - Compression distance (0.0 to 1.0)

## The 12 Experiments

| Exp | Name | Tests | Status |
|-----|------|-------|--------|
| **EXP-01** | Address Uniqueness | Zero hash collisions | ✅ PASS |
| **EXP-02** | Retrieval Efficiency | Sub-millisecond retrieval | ✅ PASS |
| **EXP-03** | Dimension Necessity | All 7 dimensions required | ✅ PASS |
| **EXP-04** | Fractal Scaling | Consistency at 1M+ scale | ✅ PASS |
| **EXP-05** | Compression/Expansion | Lossless encoding | ✅ PASS |
| **EXP-06** | Entanglement Detection | Semantic relationships | ✅ PASS |
| **EXP-07** | LUCA Bootstrap | Full system reconstruction | ✅ PASS |
| **EXP-08** | RAG Integration | Storage compatibility | ✅ PASS |
| **EXP-09** | Concurrency | Thread-safe queries | ✅ PASS |
| **EXP-10** | Bob the Skeptic | Anti-hallucination | ✅ PASS |
| **EXP-11** | Dimension Cardinality | Optimal dimension count analysis | ✅ PASS |
| **EXP-12** | Benchmark Comparison | STAT7 vs. common systems | ✅ PASS |

## EXP-01: Address Uniqueness Test

**Status**: ✅ PASS (Publication Ready)  
**Confidence**: 99.9%  
**Sample Size**: 10,000 bit-chains  

### Quick Summary

EXP-01 validates that every bit-chain in STAT7 space receives a unique address with zero hash collisions. Using SHA-256 hashing of canonical serialization, we tested 10,000 randomly generated bit-chains across 10 iterations and detected **zero collisions**, achieving a 100% uniqueness rate.

### Key Results

- **Total Bit-Chains Tested**: 10,000
- **Unique Addresses**: 10,000
- **Collisions Detected**: 0
- **Collision Rate**: 0.0%
- **Success Rate**: 100% (10/10 iterations passed)

### Documentation

- **[Methodology](docs/EXP01_METHODOLOGY.md)** - Detailed experimental design and statistical analysis
- **[Results Tables](docs/EXP01_RESULTS_TABLES.md)** - Complete iteration-by-iteration results
- **[Reproducibility Guide](docs/EXP01_REPRODUCIBILITY.md)** - Step-by-step reproduction instructions
- **[Peer Review Guide](docs/EXP01_PEER_REVIEW_GUIDE.md)** - Checklist for reviewers
- **[Publication Checklist](docs/EXP01_PUBLICATION_CHECKLIST.md)** - Publication readiness tracking
- **[Executive Summary](docs/EXP01_SUMMARY.md)** - High-level overview and conclusions

### Running EXP-01

```bash
# Run all experiments (includes EXP-01)
python -m fractalstat.stat7_experiments

# Archive results with metadata
python scripts/archive_exp01_results.py

# Generate figures (requires matplotlib)
python scripts/generate_exp01_figures.py
```

### Citation

If you use EXP-01 results in your research, please cite:

```bibtex
@software{fractalstat_exp01,
  title = {FractalStat EXP-01: Address Uniqueness Test},
  author = {[Authors]},
  year = {2024},
  version = {1.0.0},
  url = {https://gitlab.com/tiny-walnut-games/fractalstat}
}
```

## Quick Start

```bash
# Build the package
python copy_and_transform.py

# Install
pip install -e .

# Run experiments
python -m fractalstat.stat7_experiments
```

## Experiment Configuration

FractalStat uses **feature flags** to configure experiments. This allows you to:
- Run experiments with different parameters without code changes
- Use environment-specific configurations (dev, ci, production)
- Ensure reproducibility by locking configuration for publication

### Configuration Files

- `fractalstat/config/experiments.toml` - Default configuration for all experiments
- `fractalstat/config/experiments.dev.toml` - Development overrides (quick modes, smaller samples)
- `fractalstat/config/experiments.ci.toml` - CI/CD overrides (balanced for pipeline speed)

### Using Different Environments

```bash
# Use development config (fast iteration)
export FRACTALSTAT_ENV=dev
python -m fractalstat.stat7_experiments

# Use CI config (balanced testing)
export FRACTALSTAT_ENV=ci
python -m fractalstat.exp04_fractal_scaling

# Use production config (full validation)
export FRACTALSTAT_ENV=production
python -m fractalstat.exp05_compression_expansion
```

### Example Configuration

```toml
# fractalstat/config/experiments.toml
[experiments]
enabled = ["EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05", 
           "EXP-06", "EXP-07", "EXP-08", "EXP-09", "EXP-10",
           "EXP-11", "EXP-12"]

[experiments.EXP-01]
name = "Address Uniqueness Test"
sample_size = 1000
iterations = 10

[experiments.EXP-04]
name = "Fractal Scaling"
quick_mode = true
scales = [1000, 10000, 100000]
```

### Programmatic Access

```python
from fractalstat.config import ExperimentConfig

config = ExperimentConfig()

# Check if experiment is enabled
if config.is_enabled("EXP-01"):
    sample_size = config.get("EXP-01", "sample_size", 1000)
    iterations = config.get("EXP-01", "iterations", 10)
    # Run experiment...
```

For more details, see `fractalstat/config/feature_flags.py`.

## License

MIT License
