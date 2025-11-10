# FractalStat Quick Start

## Build and Install

```bash
cd fractalstat-package
python copy_and_transform.py
pip install -e .
```

## Run Experiments

```bash
# Phase 1
python -m fractalstat.stat7_experiments

# Individual experiments
python -m fractalstat.exp04_fractal_scaling
python -m fractalstat.exp06_entanglement_detection
```

## Use as Library

```python
from fractalstat import BitChain, Coordinates

bc = BitChain(...)
address = bc.compute_address()
```
