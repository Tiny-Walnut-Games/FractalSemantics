# Operations: CI Doc Checklist

Run this checklist during release prep and major workflow changes.

- Verify command examples still execute (`experiment_runner.py`, `comprehensive_experiment_analysis.py`).
- Verify analysis flags are documented (`--refresh`, `--wipe-history`, `--wipe-archive`).
- Verify README + QUICKSTART are in sync with CLI behavior.
- Verify GitBook archive pages are updated when public CLI/API changes.
- Verify deleted-doc references are removed.

Suggested checks:

```bash
git grep -n "comprehensive_experiment_analysis.py" -- "*.md"
git grep -n "experiment_runner.py" -- "*.md"
```
