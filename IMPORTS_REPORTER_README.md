# Imports Reporter

`imports_reporter.py` is a **general-purpose Python dependency audit tool**.

It is not FractalSemantics-specific: you can point it at any Python project to:

- list imported top-level modules,
- compare imports against a `requirements.txt` (including nested `-r` includes),
- highlight likely missing/unused dependencies,
- output in `json`, `text`, or `csv`.

## Features

- AST-based import detection (handles `import a, b`, aliases, and `from x import y`)
- Optional requirements comparison
- Module→package map support (`.imports_reporter_module_map.json`)
- Local-module and stdlib filtering
- Python 3.9-compatible stdlib fallback detection
- Sortable text/CSV output by module name, import count, or file count

## Quick Start

From the repository root:

### Linux / WSL / macOS

```bash
python3 imports_reporter.py . \
  --requirements ./requirements.txt \
  --format json \
  --output ./results/imports_report.json
```

### Windows (PowerShell)

```powershell
python .\imports_reporter.py . `
  --requirements .\requirements.txt `
  --format json `
  --output .\results\imports_report.json
```

## Useful Variants

### Text report sorted by highest usage

```bash
python3 imports_reporter.py . \
  --requirements ./requirements.txt \
  --format text \
  --sort-by import_count \
  --output ./results/imports_report_by_usage.txt
```

### CSV report sorted by number of files importing each module

```bash
python3 imports_reporter.py . \
  --format csv \
  --sort-by file_count \
  --output ./results/imports_report.csv
```

### Unlimited scan depth

```bash
python3 imports_reporter.py . --max-depth -1
```

## CLI Reference

```text
usage: imports_reporter.py [project_path]
  --output, -o
  --format, -f {json,csv,text}
  --sort-by {module,import_count,file_count}
  --exclude-dir <name>                (repeatable)
  --include-relative
  --requirements <path>
  --include-stdlib
  --include-local
  --ignore-module <name>              (repeatable)
  --max-depth <int>                   (-1 = unlimited)
  --show-parse-warnings
  --module-map <path>
  --create-module-map-template
  --force-module-map-template
```

## Notes

- On WSL/Linux, prefer forward-slash paths (`./requirements.txt`), not `\.\requirements.txt`.
- If `python` is not available on Ubuntu, use `python3`.
- `--include-stdlib` and `--include-local` disable filtering; by default both are filtered out in requirements comparison.
- For Python 3.9, stdlib filtering uses a safe fallback detector (builtins + stdlib path scan).

## Module Map Template

Generate template:

```bash
python3 imports_reporter.py . --create-module-map-template
```

Then edit `.imports_reporter_module_map.json` to map import names to package names (example: `"sklearn": "scikit-learn"`).
