# Python API Surface (Public/Supported)

Current stable command-entry surfaces:

- `fractalsemantics/experiment_runner.py` CLI
- `comprehensive_experiment_analysis.py` CLI

Python package entry points (from `pyproject.toml`):

- `fractalsemantics`
- `fractalsemantics-experiments`
- `fractalsemantics-runner`

Programmatic configuration access:

```python
from fractalsemantics.config import ExperimentConfig

config = ExperimentConfig()
if config.is_enabled("EXP-01"):
    sample_size = config.get("EXP-01", "sample_size", 1000)
```

Compatibility guidance:

- Treat internal experiment module internals as implementation details unless explicitly documented here.
- Prefer CLI + configuration files for stable automation.
