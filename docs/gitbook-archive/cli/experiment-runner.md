# CLI Reference: experiment_runner.py

Primary command:

```bash
python fractalsemantics/experiment_runner.py [targets] [flags]
```

Common targets:

- `--all`
- Specific IDs: `EXP-01 EXP-03 EXP-21`

Core flags:

- `--quick` / `--full`
- `--serial` or `--sequential` / `--parallel`
- `--format=text|json`
- `--repro-runs=N`
- `--softcopy=true|false`

Examples:

```bash
python fractalsemantics/experiment_runner.py --all --full --format=text
python fractalsemantics/experiment_runner.py EXP-01 EXP-03 --quick --serial --format=text
python fractalsemantics/experiment_runner.py EXP-13 --full --repro-runs=3 --format=text
python fractalsemantics/experiment_runner.py EXP-03 --quick --softcopy=false --format=text
```

Notes:

- `--softcopy=false` keeps terminal output but disables persisted artifacts.
- `--repro-runs` controls explicit reruns for reproducibility checks.
