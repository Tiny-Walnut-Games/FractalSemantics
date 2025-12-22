# FractalStat

**Complete validation suite for the STAT7 7-dimensional addressing system**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FractalStat is a comprehensive research package containing **12 validation experiments** that prove the STAT7 addressing system works at scale. STAT7 is a 7-dimensional coordinate system for uniquely addressing data in fractal information spaces.

## Quick Start

The easiest way to get started:

```bash
# Automated installation (handles platform detection)
python install.py

# Or manual installation
pip install -e .
python -m fractalstat.fractalstat_experiments
```

## What is STAT7?

The STAT7 addressing system uses 7 dimensions to uniquely identify any entity in fractal information spaces:

- **Realm**: Domain/type classification
- **Lineage**: Generation from LUCA (Last Universal Common Ancestor)
- **Adjacency**: Relational neighbors (graph connections)
- **Horizon**: Lifecycle stage
- **Luminosity**: Activity level (0-100)
- **Polarity**: Resonance/affinity type
- **Dimensionality**: Fractal depth/detail level

## The 12 Experiments

| Exp | Name | Status | Description |
|-----|------|--------|-------------|
| **EXP-01** | Geometric Collision Resistance | ✅ PASS | Zero hash collisions at scale |
| **EXP-02** | Retrieval Efficiency | ✅ PASS | Sub-millisecond lookups |
| **EXP-03** | Coordinate Entropy | ✅ PASS | Entropy contribution per dimension |
| **EXP-04** | Fractal Scaling | ✅ PASS | Consistency at 1M+ scale |
| **EXP-05** | Compression/Expansion | ✅ PASS | Lossless encoding |
| **EXP-06** | Entanglement Detection | ✅ PASS | Semantic relationships |
| **EXP-07** | LUCA Bootstrap | ✅ PASS | Full system reconstruction |
| **EXP-08** | RAG Integration | ✅ PASS | Storage compatibility |
| **EXP-09** | Concurrency | ✅ PASS | Thread-safe queries |
| **EXP-10** | Bob the Skeptic | ✅ PASS | Anti-hallucination validation |
| **EXP-11** | Dimension Cardinality | ✅ PASS | Optimal dimension analysis |
| **EXP-12** | Benchmark Comparison | ✅ PASS | STAT7 vs. common systems |

## Installation

### Easy Installation (Recommended)

```bash
# Clone repository
git clone https://gitlab.com/tiny-walnut-games/fractalstat.git
cd fractalstat

# Run automated installer (detects platform and installs dependencies)
python install.py
```

### Manual Installation

```bash
# Install with pip
pip install -e .

# For development
pip install -e ".[dev]"
```

### Raspberry Pi Installation

```bash
# Automated installer handles Raspberry Pi detection
python install.py

# Or manual lightweight installation
python install.py --minimal
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions for all platforms.

## Running Experiments

```bash
# Run all experiments (production mode)
python -m fractalstat.fractalstat_experiments

# Fast development mode (smaller samples)
export FRACTALSTAT_ENV=dev
python -m fractalstat.fractalstat_experiments

# Run individual experiment
python -m fractalstat.exp01_geometric_collision

# Use launcher script (created by installer)
./run_experiments.sh dev
```

## Configuration

FractalStat supports multiple environments:

- **Production**: Full validation with large sample sizes
- **Development**: Faster iteration with smaller samples
- **CI**: Balanced for automated testing

Configure via environment variables or TOML files in `fractalstat/config/`.

## Docker Support

```bash
# Build and run
docker build -t fractalstat .
docker run --rm fractalstat

# Development with volume mounting
docker run -it --rm -v $(pwd):/app fractalstat bash
```

## Documentation

- **[Installation Guide](INSTALL.md)** - Complete setup instructions for all platforms
- **[Experiment Documentation](docs/)** - Detailed methodology for each experiment
- **[API Reference](fractalstat/)** - Code documentation and examples

## Key Features

- ✅ **Zero Collision Guarantee**: Mathematically proven geometric collision resistance
- ✅ **Platform Agnostic**: Runs on x86, ARM, Apple Silicon, and Raspberry Pi
- ✅ **Memory Efficient**: Configurable sample sizes for resource-constrained systems
- ✅ **Production Ready**: Comprehensive error handling and logging
- ✅ **Extensively Tested**: 12 validation experiments with 100% pass rate
- ✅ **Open Source**: MIT licensed, fully reproducible research

## System Requirements

- **Python**: 3.9+
- **RAM**: 2GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Platforms**: Linux, macOS, Windows, Raspberry Pi

## Citation

If you use FractalStat in your research:

```bibtex
@software{fractalstat,
  title = {FractalStat: Complete Validation Suite for STAT7 Addressing},
  author = {Tiny Walnut Games},
  year = {2024},
  url = {https://gitlab.com/tiny-walnut-games/fractalstat},
  license = {MIT}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitLab Issues](https://gitlab.com/tiny-walnut-games/fractalstat/-/issues)
- **Documentation**: See `docs/` directory for experiment details
- **Community**: Join discussions in the GitLab repository

---

**🚀 Ready to explore fractal information spaces? Let's get started!**
