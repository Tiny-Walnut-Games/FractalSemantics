# FractalStat Installation Guide

**Complete validation suite for the STAT7 7-dimensional addressing system**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

For most users, the easiest installation is:

```bash
# Clone the repository
git clone https://gitlab.com/tiny-walnut-games/fractalstat.git
cd fractalstat

# Automated installation (recommended - handles platform detection)
python install.py

# Or manual installation
pip install -e .

# Run all experiments
python -m fractalstat.fractalstat_experiments
```

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 2GB (4GB recommended)
- **Disk**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.11+
- **RAM**: 8GB+
- **Disk**: 10GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster ML operations)

## Installation Methods

### Method 1: pip Install (Recommended)

```bash
# Install core dependencies
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"

# Verify installation
python -c "import fractalstat; print('FractalStat installed successfully!')"
```

### Method 2: Conda Install

```bash
# Create a new conda environment
conda create -n fractalstat python=3.11
conda activate fractalstat

# Install with conda
conda install pip
pip install -e .

# For ML dependencies (optional)
conda install pytorch torchvision torchaudio -c pytorch
```

### Method 3: Docker Install (Cross-Platform)

```bash
# Build Docker image
docker build -t fractalstat .

# Run experiments
docker run --rm fractalstat python -m fractalstat.fractalstat_experiments

# For development with volume mounting
docker run -it --rm -v $(pwd):/app fractalstat bash
```

## Raspberry Pi Installation

### Raspberry Pi 4/5 Setup (64-bit)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ (if not already installed)
sudo apt install python3.11 python3.11-venv python3.11-pip

# Install system dependencies for PyTorch
sudo apt install libopenblas-dev libblas-dev liblapack-dev libatlas-base-dev gfortran

# Create virtual environment
python3.11 -m venv fractalstat_env
source fractalstat_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for ARM64 (CPU only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install FractalStat
git clone https://gitlab.com/tiny-walnut-games/fractalstat.git
cd fractalstat
pip install -e .
```

### Raspberry Pi Troubleshooting

#### Memory Issues
If you encounter memory errors:

```bash
# Use development config (smaller samples)
export FRACTALSTAT_ENV=dev
python -m fractalstat.fractalstat_experiments

# Or run individual lightweight experiments
python -m fractalstat.exp01_geometric_collision
```

#### PyTorch Installation Issues
```bash
# If PyTorch installation fails, try the nightly build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Or install without PyTorch (limited functionality)
pip install -e . --no-deps
pip install pydantic numpy fastapi uvicorn click tomli
```

#### Alternative: Use Pre-built Wheels
```bash
# For Raspberry Pi OS 64-bit
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

### Raspberry Pi Performance Notes

- **Expected Performance**: Experiments run 2-5x slower than x86_64 systems
- **Memory Usage**: Monitor with `htop` or `free -h`
- **Storage**: Use external SSD for better performance
- **Cooling**: Ensure adequate cooling for long-running experiments

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt install python3 python3-pip python3-venv build-essential

# Install FractalStat
pip install -e .
```

### macOS

```bash
# Using Homebrew
brew install python@3.11

# Install FractalStat
pip install -e .

# For Apple Silicon (M1/M2)
pip install torch torchvision torchaudio
```

### Windows

```bash
# Install Python 3.11+ from python.org
# Open PowerShell as Administrator

# Install FractalStat
pip install -e .

# For GPU support (NVIDIA only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Google Colab

```python
# In a Colab notebook
!git clone https://gitlab.com/tiny-walnut-games/fractalstat.git
%cd fractalstat
!pip install -e .

# Run experiments
!python -m fractalstat.fractalstat_experiments
```

## Lightweight Installation (Minimal Dependencies)

For systems with limited resources or when you only need basic functionality:

```bash
# Install only core dependencies (no ML)
pip install pydantic>=2.0.0 numpy>=1.20.0 click>=8.1.0 tomli>=2.0.0

# Manual installation
pip install -e . --no-deps
pip install pydantic numpy click tomli

# Run basic experiments (no embeddings/LLM features)
python -m fractalstat.exp01_geometric_collision
```

## Configuration

### Environment Variables

```bash
# Set environment (dev/ci/production)
export FRACTALSTAT_ENV=dev

# Enable CUDA (if available)
export USE_CUDA=1

# Disable GPU acceleration
export USE_CUDA=0
```

### Configuration Files

FractalStat uses TOML configuration files in `fractalstat/config/`:

- `experiments.toml` - Production configuration
- `experiments.dev.toml` - Development (smaller samples, faster)
- `experiments.ci.toml` - CI/CD configuration

## Dependency Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic` | >=2.0.0 | Data validation and serialization |
| `numpy` | >=1.20.0 | Numerical computing |
| `click` | >=8.1.0 | Command-line interface |
| `tomli` | >=2.0.0 | TOML parsing (Python <3.11) |

### ML Dependencies (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | PyTorch framework |
| `transformers` | >=4.30.0 | Hugging Face models |
| `sentence-transformers` | >=2.2.0 | Text embeddings |
| `fastapi` | >=0.104.0 | Web API framework |
| `uvicorn` | >=0.24.0 | ASGI server |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=7.0.0 | Testing framework |
| `black` | >=22.0.0 | Code formatting |
| `ruff` | >=0.1.0 | Fast linter |
| `mypy` | >=1.0.0 | Type checking |

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Memory Errors
```bash
# Use development configuration
export FRACTALSTAT_ENV=dev

# Run with reduced batch size
python -m fractalstat.exp08_llm_integration --batch-size 1
```

#### PyTorch CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
export USE_CUDA=0
```

#### Permission Errors
```bash
# Use virtual environment
python -m venv fractalstat_env
source fractalstat_env/bin/activate

# Or install with --user
pip install --user -e .
```

### Getting Help

1. **Check the logs**: Look for error messages in terminal output
2. **Verify Python version**: `python --version` should be 3.9+
3. **Check dependencies**: `pip list` to see installed packages
4. **Test basic import**: `python -c "import fractalstat"`

### Performance Optimization

```bash
# Use development config for faster testing
export FRACTALSTAT_ENV=dev

# Disable unnecessary features
export DISABLE_EMBEDDINGS=1
export DISABLE_LLM=1

# Monitor resources
python -m fractalstat.fractalstat_experiments --profile
```

## Running Experiments

### All Experiments
```bash
python -m fractalstat.fractalstat_experiments
```

### Individual Experiments
```bash
# Geometric collision test
python -m fractalstat.exp01_geometric_collision

# Retrieval efficiency
python -m fractalstat.exp02_retrieval_efficiency

# Coordinate entropy
python -m fractalstat.exp03_coordinate_entropy
```

### Development Mode (Faster)
```bash
export FRACTALSTAT_ENV=dev
python -m fractalstat.fractalstat_experiments
```

## Docker Development

### Development Container
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .

CMD ["python", "-m", "fractalstat.fractalstat_experiments"]
```

### Raspberry Pi Docker
```dockerfile
FROM balenalib/raspberrypi3-64-python:latest

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "-m", "fractalstat.exp01_geometric_collision"]
```

## Testing Installation

### Basic Test
```bash
python -c "
import fractalstat
from fractalstat.config import ExperimentConfig
config = ExperimentConfig()
print('✅ FractalStat installed successfully!')
print(f'✅ Environment: {config.get_environment()}')
"
```

### Full Test Suite
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_fractalstat_experiments.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup instructions.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitLab Issues](https://gitlab.com/tiny-walnut-games/fractalstat/-/issues)
- **Documentation**: [FractalStat Docs](docs/)
- **Experiments**: See individual experiment documentation in `docs/`

---

**Happy experimenting with FractalStat! 🚀**
