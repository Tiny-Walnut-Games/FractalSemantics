---
description: Repository Information Overview
alwaysApply: true
---

# FractalSemantics Information

## Summary
FractalSemantics is a complete validation suite for the FractalSemantics 7-dimensional addressing system. It provides experimental frameworks for testing address uniqueness, retrieval efficiency, dimension necessity, compression/expansion, entanglement detection, and RAG (Retrieval-Augmented Generation) integration. The project implements Phase 1 Doctrine validation with security enums, canonical serialization, and recovery capabilities.

## Structure
- **fractalsemantics/**: Main package directory containing core FractalSemantics implementation and experiments
  - **fractalsemantics_entity.py**: Core entity definitions for the FractalSemantics addressing system
  - **fractalsemantics_experiments.py**: Phase 1 validation experiments (EXP-01, EXP-02, EXP-03)
  - **fractalsemantics_rag_bridge.py**: RAG integration bridge implementation
  - **exp04_fractal_scaling.py**: Fractal address scaling experiments
  - **exp05_compression_expansion.py**: Address compression/expansion testing
  - **exp06_entanglement_detection.py**: Entanglement detection algorithms
  - **exp07_luca_bootstrap.py**: Bootstrap validation experiments
  - **exp08_rag_integration.py**: RAG system integration tests
  - **exp09_concurrency.py**: Concurrent addressing operations
  - **bob_stress_test.py**: Stress testing suite
- **copy_and_transform.py**: Build-time script for package transformation
- **pyproject.toml**: Project metadata and dependencies configuration
- **requirements.txt**: Python dependencies list

## Language & Runtime
**Language**: Python  
**Version**: 3.9, 3.10, 3.11  
**Build System**: setuptools/wheel  
**Package Manager**: pip  

## Dependencies

**Python Version**: 3.11+
**Terminal Support**: ANSI escape codes supported (Windows PowerShell, Linux Terminal) 
**Operating Systems**: Windows, macOS, Linux 
**Note**: all terminals run as user mode and that means nothing is in path at the start of your turn. You must add python and all dependencies to PATH manually before beginning work.

**Main Dependencies**:
- pydantic ≥2.0.0 (data validation)
- numpy ≥1.20.0 (numerical computations)
- torch ≥2.0.0 (deep learning)
- transformers ≥4.30.0 (NLP models)
- sentence-transformers ≥2.2.0 (embedding models)
- fastapi ≥0.104.0 (API framework)
- uvicorn ≥0.24.0 (ASGI server)
- click ≥8.1.0 (CLI framework)

**Development Dependencies**:
- pytest ≥7.0.0 (testing framework)
- pytest-asyncio ≥0.21.0 (async test support)
- black ≥22.0.0 (code formatting)

## Build & Installation
```bash
cd fractalsemantics
python copy_and_transform.py
pip install -e .
```

## Main Entry Points
**Phase 1 Validation**:
```bash
python -m fractalsemantics.fractalsemantics_experiments
```

**Individual Experiments**:
```bash
python -m fractalsemantics.exp04_fractal_scaling
python -m fractalsemantics.exp05_compression_expansion
python -m fractalsemantics.exp06_entanglement_detection
python -m fractalsemantics.exp07_luca_bootstrap
python -m fractalsemantics.exp08_rag_integration
python -m fractalsemantics.exp09_concurrency
```

**Library Usage**:
```python
from fractalsemantics import BitChain, Coordinates
bc = BitChain(...)
address = bc.compute_address()
```

## Testing
**Framework**: pytest  
**Test Location**: tests/ (configured in pyproject.toml)  
**Configuration File**: pyproject.toml [tool.pytest.ini_options]  
**Run Command**:
```bash
pytest
```

## CI/CD
**Platform**: GitLab CI/CD  
**Configuration**: .gitlab-ci.yml  
**Pipeline Stages**: test, secret-detection  
**Security Checks**: SAST, Secret Detection enabled  
