"""Fail CI when public CLI documentation drifts from expected flag coverage."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

README = PROJECT_ROOT / "README.md"
QUICKSTART = PROJECT_ROOT / "QUICKSTART.md"
RUNNER_DOC = PROJECT_ROOT / "docs" / "gitbook-archive" / "cli" / "experiment-runner.md"
ANALYSIS_DOC = PROJECT_ROOT / "docs" / "gitbook-archive" / "cli" / "analysis-cli.md"
SUMMARY_DOC = PROJECT_ROOT / "docs" / "gitbook-archive" / "SUMMARY.md"

REQUIRED = {
    README: {
        "analysis": [
            "comprehensive_experiment_analysis.py",
            "--refresh",
            "--wipe-history",
            "--wipe-archive",
        ],
        "runner": [
            "fractalsemantics/experiment_runner.py",
            "--repro-runs",
            "--softcopy",
        ],
    },
    ANALYSIS_DOC: {
        "analysis": [
            "--refresh",
            "--wipe-history",
            "--wipe-archive",
        ],
    },
    RUNNER_DOC: {
        "runner": [
            "--repro-runs",
            "--softcopy",
            "--quick",
            "--full",
            "--format=text|json",
        ],
    },
    QUICKSTART: {
        "scope": [
            "full public CLI/API scope",
            "docs/gitbook-archive/SUMMARY.md",
        ],
    },
    SUMMARY_DOC: {
        "nav": [
            "cli/experiment-runner.md",
            "cli/analysis-cli.md",
            "api/python-api-surface.md",
        ],
    },
}


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required docs file: {path}")
    return path.read_text(encoding="utf-8")


def main() -> int:
    failures: list[str] = []

    for path, checks in REQUIRED.items():
        try:
            content = _read(path)
        except Exception as exc:
            failures.append(str(exc))
            continue

        for category, phrases in checks.items():
            for phrase in phrases:
                if phrase not in content:
                    failures.append(f"{path}: missing {category} token '{phrase}'")

    if failures:
        print("[docs-sync] FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[docs-sync] OK: CLI/API documentation coverage checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
