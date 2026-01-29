# FractalStat Quick Start

## Prerequisites

- Python 3.9+
- pip (Python package installer)

### ARM/Raspberry Pi Notes

FractalStat works on ARM architectures (like Raspberry Pi), but some dependencies may require additional setup:

- **PyTorch**: Use ARM-compatible wheels or install from source
- **Transformers/SentenceTransformers**: Generally work on ARM, but may be slower without GPU acceleration
- **Memory**: Some experiments may require significant RAM (4GB+ recommended)

For Raspberry Pi, you may need to install PyTorch specifically for ARM:

```bash
# For Raspberry Pi (ARM64)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
pip install -e .
```

## Build and Install

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Run Experiments

```bash
# Phase 1 validation experiments
python -m fractalstat.fractalstat_experiments

# Individual experiments
python -m fractalstat.exp04_fractal_scaling
python -m fractalstat.exp06_entanglement_detection
```

## Use as Library

```python
from fractalstat import BitChain, Coordinates

# Create a bit-chain
bc = BitChain(
    id="example",
    entity_type="concept",
    realm="data",
    coordinates=Coordinates(
        realm="data",
        lineage=1,
        adjacency=[],
        horizon="genesis",
        resonance=0.5,
        velocity=0.0,
        density=0.5
    ),
    created_at="2024-01-01T00:00:00.000Z",
    state={"value": 42}
)

# Compute FractalStat address
address = bc.compute_address()
print(f"FractalStat Address: {address}")
```

## Configuration

FractalStat uses environment-specific configurations:

```bash
# Development mode (fast iteration)
export FRACTALSTAT_ENV=dev
python -m fractalstat.fractalstat_experiments

# CI mode (balanced testing)
export FRACTALSTAT_ENV=ci
python -m fractalstat.fractalstat_experiments

# Production mode (full validation)
export FRACTALSTAT_ENV=production
python -m fractalstat.fractalstat_experiments
```
