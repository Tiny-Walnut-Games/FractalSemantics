# FractalSemantics - FractalSemantics Validation Experiments

- **A complete validation suite for the FractalSemantics 8-dimensional addressing system**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> 🔬 **Recent Discovery (Nov 2025):** EXP-11 testing confirms **8 dimensions are optimal** and superior to the original 7-dimension design. See [!5](https://gitlab.com/tiny-walnut-games/fractalsemantics/-/merge_requests/5) for details. EXP-01 validation results remain valid as they are dimension-count agnostic. FractalSemantics implementation complete.

## What is FractalSemantics?

FractalSemantics is a research package containing **12 validation experiments** that prove the FractalSemantics (previously known as FractalSemantics internally) addressing system works at scale. FractalSemantics expands FractalSemantics from a 7D to an 8-dimensional coordinate system for uniquely addressing data in fractal information spaces.

**The 8 Dimensions:**

- **Realm** - Domain classification (data, narrative, system, etc.)
- **Lineage** - Generation from LUCA (Last Universal Common Ancestor)
- **Temperature** - Thermal activity level (0.0 to abs(velocity) * density)
- **Adjacency** - Relational neighbors (graph connections)
- **Horizon** - Lifecycle stage (genesis, emergence, peak, decay, crystallization)
- **Resonance** - Charge/alignment (-1.0 to 1.0)
- **Velocity** - Rate of change
- **Density** - Compression distance (0.0 to 1.0)
- **Alignment** - Value based on alignment map

## The 12 Experiments

| Exp | Name | Tests | Status |
|-----|------|-------|--------|
| **EXP-01** | Geometric Collisions | Zero collisions over 3D | [Success] PASS |
| **EXP-02** | Retrieval Efficiency | Sub-millisecond retrieval | [Success] PASS |
| **EXP-03** | Coordinate Space Entropy | Entropy contribution per dimension | [Success] PASS |
| **EXP-04** | Fractal Scaling | Consistency at 1M+ scale | [Success] PASS |
| **EXP-05** | Compression/Expansion | Lossless encoding | [Success] PASS |
| **EXP-06** | Entanglement Detection | Semantic relationships | [Success] PASS |
| **EXP-07** | LUCA Bootstrap | Full system reconstruction | [Success] PASS |
| **EXP-08** | RAG Integration | Storage compatibility | [Success] PASS |
| **EXP-09** | Concurrency | Thread-safe queries | [Success] PASS |
| **EXP-10** | Bob the Skeptic | Anti-hallucination | [Success] PASS |
| **EXP-11** | Dimension Cardinality | Optimal dimension count analysis | [Success] PASS |
| **EXP-12** | Benchmark Comparison | FractalSemantics vs. common systems | [Success] PASS |

## EXP-01: Address Uniqueness Test

**Status**: [Success] PASS (Publication Ready)  
**Confidence**: 99.9%  
**Sample Size**: 10,000 bit-chains  

### Quick Summary

EXP-01 has been rewritten to illustrate how increasing the number of dimensions naturally eliminates concerns of collision. While our SHA-256 hashing of canonical serialization already guarantees zero collisions, this test proves that SHA256 is a security choice and not our crutch for collision-free addressing.

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
python -m fractalsemantics.fractalsemantics_experiments

# Archive results with metadata
python scripts/archive_exp01_results.py

# Generate figures (requires matplotlib)
python scripts/generate_exp01_figures.py
```

### Citation

If you use EXP-01 results in your research, please cite:

```bibtex
@software{fractalsemantics_exp01,
  title = {FractalSemantics EXP-01: Address Uniqueness Test},
  author = {[Authors]},
  year = {2024},
  version = {1.0.0},
  url = {https://gitlab.com/tiny-walnut-games/fractalsemantics}
}
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Run experiments
python -m fractalsemantics.fractalsemantics_experiments
```

### ARM/Raspberry Pi Setup

FractalSemantics works on ARM architectures, but PyTorch installation may require special handling:

```bash
# For Raspberry Pi (ARM64) - Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
pip install -e .

# Run experiments (may be slower on ARM without GPU)
python -m fractalsemantics.fractalsemantics_experiments
```

**ARM Considerations:**

- Some experiments (EXP-08 LLM integration) may be slower without GPU acceleration
- Memory usage can be high - 4GB+ RAM recommended
- All core FractalSemantics functionality works identically across architectures

## Experiment Configuration

FractalSemantics uses **feature flags** to configure experiments. This allows you to:

- Run experiments with different parameters without code changes
- Use environment-specific configurations (dev, ci, production)
- Ensure reproducibility by locking configuration for publication

### Configuration Files

- `fractalsemantics/config/experiments.toml` - Default configuration for all experiments
- `fractalsemantics/config/experiments.dev.toml` - Development overrides (quick modes, smaller samples)
- `fractalsemantics/config/experiments.ci.toml` - CI/CD overrides (balanced for pipeline speed)

### Using Different Environments

```bash
# Use development config (fast iteration)
export FRACTALSEMANTICS_ENV=dev
python -m fractalsemantics.fractalsemantics_experiments

# Use CI config (balanced testing)
export FRACTALSEMANTICS_ENV=ci
python -m fractalsemantics.exp04_fractal_scaling

# Use production config (full validation)
export FRACTALSEMANTICS_ENV=production
python -m fractalsemantics.exp05_compression_expansion
```

### Example Configuration

```toml
# fractalsemantics/config/experiments.toml
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
from fractalsemantics.config import ExperimentConfig

config = ExperimentConfig()

# Check if experiment is enabled
if config.is_enabled("EXP-01"):
    sample_size = config.get("EXP-01", "sample_size", 1000)
    iterations = config.get("EXP-01", "iterations", 10)
    # Run experiment...
```

For more details, see `fractalsemantics/config/feature_flags.py`.

## License

MIT License
