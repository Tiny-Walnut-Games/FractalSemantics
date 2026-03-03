"""Fail CI when documented CLI command shapes drift from live CLI surfaces."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

README = PROJECT_ROOT / "README.md"
QUICKSTART = PROJECT_ROOT / "QUICKSTART.md"
RUNNER_DOC = PROJECT_ROOT / "docs" / "gitbook-archive" / "cli" / "experiment-runner.md"
ANALYSIS_DOC = PROJECT_ROOT / "docs" / "gitbook-archive" / "cli" / "analysis-cli.md"

RUNNER_SCRIPT = PROJECT_ROOT / "fractalsemantics" / "experiment_runner.py"
ANALYSIS_SCRIPT = PROJECT_ROOT / "comprehensive_experiment_analysis.py"

FLAG_PATTERN = re.compile(r"--[a-z0-9-]+")


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required docs file: {path}")
    return path.read_text(encoding="utf-8")


def _run_cli(args: list[str], expected_exit_codes: set[int]) -> str:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode not in expected_exit_codes:
        raise RuntimeError(
            f"CLI command failed unexpectedly: {' '.join(args)}\n"
            f"exit={proc.returncode}\n"
            f"output:\n{output}"
        )
    if not output:
        raise RuntimeError(f"No CLI output captured from: {' '.join(args)}")
    return output


def _extract_flags(cli_output: str) -> set[str]:
    return set(FLAG_PATTERN.findall(cli_output))


def _contains_any(content: str, aliases: set[str]) -> bool:
    return any(alias in content for alias in aliases)


def main() -> int:
    failures: list[str] = []

    docs_map = {
        README: _read(README),
        QUICKSTART: _read(QUICKSTART),
        RUNNER_DOC: _read(RUNNER_DOC),
        ANALYSIS_DOC: _read(ANALYSIS_DOC),
    }

    runner_output = _run_cli([str(RUNNER_SCRIPT)], expected_exit_codes={1})
    analysis_output = _run_cli([str(ANALYSIS_SCRIPT), "--help"], expected_exit_codes={0})

    runner_flags = _extract_flags(runner_output)
    analysis_flags = _extract_flags(analysis_output)

    required_runner_groups = [
        {"--all"},
        {"--quick"},
        {"--full"},
        {"--parallel"},
        {"--serial", "--sequential"},
        {"--format"},
        {"--repro-runs"},
        {"--softcopy"},
    ]
    required_analysis_groups = [
        {"--refresh"},
        {"--wipe-history"},
        {"--wipe-archive"},
    ]

    for group in required_runner_groups:
        if not (runner_flags & group):
            failures.append(
                f"Runner CLI missing expected public flag group: {sorted(group)}"
            )

    for group in required_analysis_groups:
        if not (analysis_flags & group):
            failures.append(
                f"Analysis CLI missing expected public flag group: {sorted(group)}"
            )

    docs_runner_groups = [
        {"--all"},
        {"--quick", "--full"},
        {"--serial", "--sequential"},
        {"--parallel"},
        {"--format"},
        {"--repro-runs"},
        {"--softcopy"},
    ]
    docs_analysis_groups = [
        {"--refresh"},
        {"--wipe-history"},
        {"--wipe-archive"},
    ]

    runner_doc_targets = [README, QUICKSTART, RUNNER_DOC]
    analysis_doc_targets = [README, QUICKSTART, ANALYSIS_DOC]

    for path in runner_doc_targets:
        content = docs_map[path]
        for group in docs_runner_groups:
            if not _contains_any(content, group):
                failures.append(f"{path}: missing runner flag group {sorted(group)}")

    for path in analysis_doc_targets:
        content = docs_map[path]
        for group in docs_analysis_groups:
            if not _contains_any(content, group):
                failures.append(f"{path}: missing analysis flag group {sorted(group)}")

    if failures:
        print("[docs-command-shape] FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[docs-command-shape] OK: CLI command shape and docs flag coverage aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
