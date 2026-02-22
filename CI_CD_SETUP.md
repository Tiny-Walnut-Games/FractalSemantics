# CI/CD Setup Guide

## Current Validation Workflow

This repository’s current experiment workflow is centered on `fractalsemantics/experiment_runner.py` and `comprehensive_experiment_analysis.py`.

### 1) Quality Gates

Run before every merge:

```bash
black --check fractalsemantics/
ruff check fractalsemantics/
mypy fractalsemantics/ --ignore-missing-imports
pytest
python scripts/check_docs_sync.py
python scripts/check_docs_command_shape.py
```

### 2) Experiment Validation

```bash
# Full suite
python fractalsemantics/experiment_runner.py --all --full --format=text

# Optional reproducibility checks
python fractalsemantics/experiment_runner.py --all --full --repro-runs=2 --format=text
```

### 3) Analysis Report

```bash
# Analyze cached result files (no reruns)
python comprehensive_experiment_analysis.py

# Refresh + analyze in one command (runner args pass through)
python comprehensive_experiment_analysis.py --refresh --all --full --serial --format=text
```

### 4) Clean/Reset Run History

```bash
# Prompt to archive history before wiping
python comprehensive_experiment_analysis.py --wipe-history --refresh --all --full --serial --format=text

# Nuclear option (deletes archive store and force-deletes history)
python comprehensive_experiment_analysis.py --wipe-archive --wipe-history --refresh --all --full --serial --format=text
```

## Artifacts

- Result JSON: `results/*.json`
- Figures: `results/figures/`
- Reports: `results/reports/`
- Comprehensive analysis report: `comprehensive_experiment_analysis.txt`

## Suggested CI Stages

1. `quality` (black, ruff, mypy)
2. `tests` (pytest)
3. `experiments` (runner full suite)
4. `analysis` (comprehensive analysis)
5. `package` (optional build/publish)

## Notes

- Prefer `--format=text` for readable CI logs.
- Use `--serial` in CI when deterministic output ordering is preferred.
- Use `--softcopy=false` when CI needs terminal output without persisted artifacts.

## Doc Retirement Checklist

Use this during release prep or major workflow changes:

1. Identify stale docs (old fix summaries, outdated status snapshots, superseded how-to files).
2. Verify active docs (`README.md`, `QUICKSTART.md`, `INSTALL.md`, `CI_CD_SETUP.md`, `GUI_README.md`) reflect current commands and flags.
3. Remove stale docs from the working tree once superseded.
4. Confirm recovery path exists via git history:

```bash
git log --diff-filter=D --name-status -- "*.md"
git log --follow -- path/to/file.md
git show <commit_sha>:path/to/file.md
```

5. Run a quick consistency sweep for dead references:

```bash
git grep -n "_FIX_SUMMARY.md\|WORKFLOW_SUMMARY\|PROJECT_STATUS_SUMMARY" -- "*.md"
```
