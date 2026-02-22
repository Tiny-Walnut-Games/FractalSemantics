# FractalSemantics Quick Start

Quickstart intentionally covers the common path.
For the full public CLI/API scope (all runner/analysis flags and behavior), see [README.md](README.md)
and the GitBook-ready archive at [docs/gitbook-archive/SUMMARY.md](docs/gitbook-archive/SUMMARY.md).

## Prerequisites

- Python 3.11+
- pip

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Run Experiments (Current Runner)

```bash
# Full suite
python fractalsemantics/experiment_runner.py --all --full --format=text

# Fast iteration run
python fractalsemantics/experiment_runner.py --all --quick --format=text

# Explicit parallel mode
python fractalsemantics/experiment_runner.py --all --full --parallel --format=text

# Specific experiments
python fractalsemantics/experiment_runner.py EXP-01 EXP-03 EXP-21 --full --serial --format=text

# Reproducibility reruns
python fractalsemantics/experiment_runner.py EXP-13 --full --repro-runs=3 --format=text

# Terminal-only mode (no persisted artifacts)
python fractalsemantics/experiment_runner.py EXP-03 --quick --softcopy=false --format=text
```

## Run Analysis (Cached by Default)

```bash
# Analyze existing results (does not rerun experiments)
python comprehensive_experiment_analysis.py

# Refresh results first, forwarding runner arguments
python comprehensive_experiment_analysis.py --refresh --all --full --serial --format=text

# Wipe history before refresh (prompts to archive vs delete)
python comprehensive_experiment_analysis.py --wipe-history --refresh --all --full --serial --format=text

# Nuclear option: delete archive store + force delete history
python comprehensive_experiment_analysis.py --wipe-archive --wipe-history --refresh --all --full --serial --format=text
```

## GUI

```bash
pip install -r gui_requirements.txt
python launch_gui.py
```
