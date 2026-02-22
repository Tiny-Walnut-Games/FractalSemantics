# CLI Reference: comprehensive_experiment_analysis.py

Primary command:

```bash
python comprehensive_experiment_analysis.py [analysis-flags] [runner-flags-after---refresh]
```

Analysis flags:

- `--refresh`
  - Runs `fractalsemantics/experiment_runner.py` before analysis.
  - Unknown args are forwarded to the runner.
- `--wipe-history`
  - Clears history artifacts before refresh/analysis.
  - Prompts archive vs delete unless forced.
- `--wipe-archive`
  - Nuclear option: deletes archive store and force-deletes wipe-history targets.

Wipe targets for `--wipe-history`:

- `results/*.json`
- `results/figures/*`
- `results/reports/*`

Examples:

```bash
# Cached analysis only (no reruns)
python comprehensive_experiment_analysis.py

# Refresh first, then analyze
python comprehensive_experiment_analysis.py --refresh --all --full --serial --format=text

# Archive-or-delete prompt before refresh
python comprehensive_experiment_analysis.py --wipe-history --refresh --all --full --serial --format=text

# Nuclear force-delete path
python comprehensive_experiment_analysis.py --wipe-archive --wipe-history --refresh --all --full --serial --format=text
```

Behavior notes:

- Without `--refresh`, extra runner args are ignored.
- Cached analysis mode avoids duplicate reruns/artifacts by default.
