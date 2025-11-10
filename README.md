# FractalStat - STAT7 Validation Experiments

**A complete validation suite for the STAT7 7-dimensional addressing system**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## What is FractalStat?

FractalStat is a research package containing **10 validation experiments** that prove the STAT7 addressing system works at scale. STAT7 is a 7-dimensional coordinate system for uniquely addressing data in fractal information spaces.

**The 7 Dimensions:**
- **Realm** - Domain classification (data, narrative, system, etc.)
- **Lineage** - Generation from LUCA (Last Universal Common Ancestor)
- **Adjacency** - Relational neighbors (graph connections)
- **Horizon** - Lifecycle stage (genesis, emergence, peak, decay, crystallization)
- **Resonance** - Charge/alignment (-1.0 to 1.0)
- **Velocity** - Rate of change
- **Density** - Compression distance (0.0 to 1.0)

## The 10 Experiments

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

## Quick Start

```bash
# Build the package
python copy_and_transform.py

# Install
pip install -e .

# Run experiments
python -m fractalstat.stat7_experiments
```

## License

MIT License
