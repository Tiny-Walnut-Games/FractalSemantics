"""Report imported Python modules across a project.

This scanner is aimed at dependency hygiene workflows (e.g., checking
requirements coverage). It parses files with ``ast`` instead of line-based
string matching, so it correctly handles:
- multi-import lines (``import a, b``)
- aliased imports (``import numpy as np``)
- nested/module imports (``from pkg.sub import x``)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pkgutil
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path
from typing import TextIO, TypedDict

DEFAULT_EXCLUDED_DIRS: set[str] = {
    ".git",
    ".history",
    ".cline",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
    "htmlcov",
}

DEFAULT_MODULE_MAP_FILENAME = ".imports_reporter_module_map.json"
DEFAULT_MODULE_MAP_TEMPLATE: dict[str, str] = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "streamlit_autorefresh": "streamlit-autorefresh",
}

class ImportDetails(TypedDict):
    import_count: int
    file_count: int
    files: list[str]


class RequirementsComparison(TypedDict):
    imported_not_in_requirements: list[str]
    requirements_not_imported: list[str]
    ignored_imports: list[str]
    ignored_local_modules: list[str]
    applied_module_map: dict[str, str]


ImportReport = dict[str, ImportDetails]
SortBy = str


def iter_sorted_modules(report: ImportReport, sort_by: SortBy) -> list[tuple[str, ImportDetails]]:
    """Return report items sorted by selected key with stable tie-breaking."""
    items = list(report.items())
    if sort_by == "import_count":
        return sorted(items, key=lambda item: (-item[1]["import_count"], item[0]))
    if sort_by == "file_count":
        return sorted(items, key=lambda item: (-item[1]["file_count"], item[0]))
    return sorted(items, key=lambda item: item[0])


def iter_python_files(
    project_path: Path,
    excluded_dirs: set[str],
    max_depth: int | None,
) -> list[Path]:
    """Return Python files under ``project_path`` while skipping excluded dirs.

    Depth is counted from ``project_path`` itself (depth 0).
    """
    python_files: list[Path] = []
    for root, dirs, files in os.walk(project_path):
        root_path = Path(root)
        try:
            depth = len(root_path.relative_to(project_path).parts)
        except ValueError:
            continue

        dirs[:] = [directory for directory in dirs if directory not in excluded_dirs]
        if max_depth is not None and depth >= max_depth:
            dirs[:] = []

        for file_name in files:
            if file_name.endswith(".py"):
                python_files.append(root_path / file_name)
    return python_files

def extract_imported_modules(
    file_path: Path,
    include_relative: bool,
    show_parse_warnings: bool = False,
) -> list[str]:
    """Extract imported module names from one Python file."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as error:
        if show_parse_warnings:
            print(f"Warning: could not read {file_path}: {error}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as error:
        if show_parse_warnings:
            print(f"Warning: syntax error in {file_path}: {error}", file=sys.stderr)
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module.split(".")[0])
            elif include_relative:
                for alias in node.names:
                    if alias.name != "*":
                        modules.append(alias.name.split(".")[0])

    return modules


def scan_imports(
    project_path: Path,
    excluded_dirs: set[str],
    include_relative: bool,
    max_depth: int | None,
    show_parse_warnings: bool = False,
) -> ImportReport:
    """Scan imports for all Python files in ``project_path``."""
    imports_map: dict[str, list[str]] = defaultdict(list)

    for py_file in iter_python_files(project_path, excluded_dirs, max_depth=max_depth):
        relative_path = str(py_file.relative_to(project_path)).replace("\\", "/")
        for module in extract_imported_modules(
            py_file,
            include_relative=include_relative,
            show_parse_warnings=show_parse_warnings,
        ):
            imports_map[module].append(relative_path)

    report: ImportReport = {}
    for module in sorted(imports_map):
        all_occurrences = imports_map[module]
        unique_files = sorted(set(all_occurrences))
        report[module] = {
            "import_count": len(all_occurrences),
            "file_count": len(unique_files),
            "files": unique_files,
        }
    return report


def normalize_requirement_name(name: str) -> str:
    """Normalize requirement/package names for comparison."""
    return name.strip().lower().replace("-", "_")


def get_stdlib_module_names(runtime_sys: object = sys) -> set[str]:
    """Return stdlib module names with a Python 3.9-compatible fallback.

    Python 3.10+ exposes ``sys.stdlib_module_names``. For Python 3.9, fall back
    to a conservative stdlib scan (builtins + top-level stdlib path modules).
    """
    stdlib_names = getattr(runtime_sys, "stdlib_module_names", None)
    if stdlib_names:
        return {normalize_requirement_name(str(name)) for name in stdlib_names}

    detected: set[str] = {
        normalize_requirement_name(str(name)) for name in getattr(runtime_sys, "builtin_module_names", ())
    }

    detected.update({"sys", "builtins", "__future__"})

    stdlib_dir_raw = sysconfig.get_paths().get("stdlib")
    if stdlib_dir_raw:
        stdlib_dir = Path(stdlib_dir_raw)
        if stdlib_dir.exists():
            try:
                for module in pkgutil.iter_modules([str(stdlib_dir)]):
                    detected.add(normalize_requirement_name(module.name))
            except OSError:
                pass

    detected.discard("site_packages")
    detected.discard("dist_packages")
    return detected


def parse_requirements_file(requirements_path: Path) -> set[str]:
    """Parse requirements-style file into normalized package names."""
    return _parse_requirements_file(requirements_path.resolve(), visited=set())


def _parse_requirements_file(requirements_path: Path, visited: set[Path]) -> set[str]:
    """Parse requirements files, following nested include directives."""
    if requirements_path in visited:
        return set()
    visited.add(requirements_path)

    packages: set[str] = set()

    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return packages

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        candidate = line.split("#", 1)[0].strip()
        if not candidate:
            continue

        include_target: str | None = None
        if candidate.startswith("-r ") or candidate.startswith("--requirement "):
            include_target = candidate.split(maxsplit=1)[1].strip()
        elif candidate.startswith("-r") and len(candidate) > 2:
            include_target = candidate[2:].strip()

        if include_target:
            include_path = (requirements_path.parent / include_target).resolve()
            packages.update(_parse_requirements_file(include_path, visited=visited))
            continue

        if candidate.startswith(("-c", "--constraint", "-e", "--editable")):
            continue
        for separator in ("==", ">=", "<=", "~=", "!=", "<", ">", ";", "["):
            if separator in candidate:
                candidate = candidate.split(separator, 1)[0].strip()
                break

        if candidate:
            packages.add(normalize_requirement_name(candidate))

    return packages


def create_module_map_template(module_map_path: Path, force: bool = False) -> bool:
    """Create a starter module-to-package mapping template file."""
    if module_map_path.exists() and not force:
        return False

    module_map_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "module_to_package": DEFAULT_MODULE_MAP_TEMPLATE,
    }
    module_map_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def load_module_map(module_map_path: Path) -> dict[str, str]:
    """Load module-to-package mapping from JSON file."""
    try:
        raw = json.loads(module_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(raw, dict) and isinstance(raw.get("module_to_package"), dict):
        source_map = raw["module_to_package"]
    elif isinstance(raw, dict):
        source_map = raw
    else:
        return {}

    normalized_map: dict[str, str] = {}
    for raw_module, raw_package in source_map.items():
        if not isinstance(raw_module, str) or not isinstance(raw_package, str):
            continue
        module_name = normalize_requirement_name(raw_module)
        package_name = normalize_requirement_name(raw_package)
        if module_name and package_name:
            normalized_map[module_name] = package_name

    return normalized_map


def compare_imports_to_requirements(
    report: ImportReport,
    requirements_packages: set[str],
    ignored_imports: set[str] | None = None,
    local_modules: set[str] | None = None,
    module_map: dict[str, str] | None = None,
) -> RequirementsComparison:
    """Build a simple module/package comparison for dependency hygiene."""
    imported_modules = {normalize_requirement_name(module) for module in report}
    effective_ignored_imports = ignored_imports or set()
    comparable_imports = imported_modules - effective_ignored_imports
    effective_module_map = module_map or {}

    mapped_imports: set[str] = set()
    applied_module_map: dict[str, str] = {}
    for module in comparable_imports:
        mapped_package = effective_module_map.get(module, module)
        mapped_imports.add(mapped_package)
        if mapped_package != module:
            applied_module_map[module] = mapped_package

    imported_not_in_requirements = sorted(mapped_imports - requirements_packages)
    requirements_not_imported = sorted(requirements_packages - mapped_imports)

    return {
        "imported_not_in_requirements": imported_not_in_requirements,
        "requirements_not_imported": requirements_not_imported,
        "ignored_imports": sorted(effective_ignored_imports & imported_modules),
        "ignored_local_modules": sorted((local_modules or set()) & imported_modules),
        "applied_module_map": dict(sorted(applied_module_map.items())),
    }


def discover_local_modules(
    project_path: Path,
    excluded_dirs: set[str],
    max_depth: int | None,
) -> set[str]:
    """Discover local module/package names reachable from scanned project files."""
    local_modules: set[str] = set()

    for py_file in iter_python_files(project_path, excluded_dirs, max_depth=max_depth):
        local_modules.add(normalize_requirement_name(py_file.stem))

    for root, dirs, files in os.walk(project_path):
        root_path = Path(root)
        try:
            depth = len(root_path.relative_to(project_path).parts)
        except ValueError:
            continue

        dirs[:] = [directory for directory in dirs if directory not in excluded_dirs]
        if max_depth is not None and depth >= max_depth:
            dirs[:] = []

        if "__init__.py" in files:
            package_name = normalize_requirement_name(root_path.name)
            if package_name:
                local_modules.add(package_name)

    return local_modules


def build_ignored_imports(
    project_path: Path,
    excluded_dirs: set[str],
    max_depth: int | None,
    include_stdlib: bool,
    include_local: bool,
    extra_ignored: list[str],
) -> tuple[set[str], set[str]]:
    """Build normalized import names to ignore in requirements comparison."""
    ignored: set[str] = set()
    local_modules: set[str] = set()

    if not include_stdlib:
        ignored.update(get_stdlib_module_names())

    if not include_local:
        local_modules = discover_local_modules(
            project_path=project_path,
            excluded_dirs=excluded_dirs,
            max_depth=max_depth,
        )
        ignored.update(local_modules)

    ignored.update(normalize_requirement_name(name) for name in extra_ignored)
    return ignored, local_modules


def write_json(
    report: ImportReport,
    output_stream: TextIO,
    requirements_comparison: RequirementsComparison | None = None,
) -> None:
    payload: ImportReport | dict[str, object]
    if requirements_comparison is None:
        payload = report
    else:
        payload = {
            "imports": report,
            "requirements_comparison": requirements_comparison,
        }

    json.dump(payload, output_stream, indent=2, sort_keys=True)
    output_stream.write("\n")


def write_csv(report: ImportReport, output_stream: TextIO, sort_by: SortBy = "module") -> None:
    writer = csv.writer(output_stream)
    writer.writerow(["Module", "ImportCount", "FileCount", "Files"])
    for module, details in iter_sorted_modules(report, sort_by=sort_by):
        writer.writerow([
            module,
            details["import_count"],
            details["file_count"],
            ", ".join(details["files"]),
        ])


def write_text(
    report: ImportReport,
    output_stream: TextIO,
    requirements_comparison: RequirementsComparison | None = None,
    sort_by: SortBy = "module",
) -> None:
    for module, details in iter_sorted_modules(report, sort_by=sort_by):
        output_stream.write(
            f"{module} (imports={details['import_count']}, files={details['file_count']}): "
            f"{', '.join(details['files'])}\n"
        )

    if requirements_comparison is None:
        return

    output_stream.write("\nRequirements comparison:\n")
    output_stream.write("- Imported but missing from requirements:\n")
    if requirements_comparison["imported_not_in_requirements"]:
        for module in requirements_comparison["imported_not_in_requirements"]:
            output_stream.write(f"  - {module}\n")
    else:
        output_stream.write("  - (none)\n")

    output_stream.write("- In requirements but not imported:\n")
    if requirements_comparison["requirements_not_imported"]:
        for package in requirements_comparison["requirements_not_imported"]:
            output_stream.write(f"  - {package}\n")
    else:
        output_stream.write("  - (none)\n")

    output_stream.write("- Ignored imports (filter applied):\n")
    if requirements_comparison["ignored_imports"]:
        for module in requirements_comparison["ignored_imports"]:
            output_stream.write(f"  - {module}\n")
    else:
        output_stream.write("  - (none)\n")

    output_stream.write("- Applied module map:\n")
    if requirements_comparison["applied_module_map"]:
        for module, package in requirements_comparison["applied_module_map"].items():
            output_stream.write(f"  - {module} -> {package}\n")
    else:
        output_stream.write("  - (none)\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan imports of a Python project.")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=os.getcwd(),
        help="Path to the Python project (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save the results (default: print to console)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "text"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["module", "import_count", "file_count"],
        default="module",
        help="Sort key for text/csv output (default: module)",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude (can be specified multiple times)",
    )
    parser.add_argument(
        "--include-relative",
        action="store_true",
        help="Include imports from relative statements like 'from . import x'",
    )
    parser.add_argument(
        "--requirements",
        help="Path to requirements file for import-vs-requirements comparison",
    )
    parser.add_argument(
        "--include-stdlib",
        action="store_true",
        help="Include Python standard library modules in requirements comparison",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Include top-level local project modules in requirements comparison",
    )
    parser.add_argument(
        "--ignore-module",
        action="append",
        default=[],
        help="Module name to ignore in requirements comparison (can be repeated)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help=(
            "Maximum directory depth to scan from project root "
            "(default: 5, use -1 for no depth limit)"
        ),
    )
    parser.add_argument(
        "--show-parse-warnings",
        action="store_true",
        help="Show parse/read warnings for files that cannot be analyzed",
    )
    parser.add_argument(
        "--module-map",
        help=(
            "Path to JSON module-to-package map file; if omitted, "
            f"{DEFAULT_MODULE_MAP_FILENAME} in project root is used when present"
        ),
    )
    parser.add_argument(
        "--create-module-map-template",
        action="store_true",
        help="Create a starter module-to-package map template and exit",
    )
    parser.add_argument(
        "--force-module-map-template",
        action="store_true",
        help="Overwrite module map template if it already exists",
    )
    return parser


def run() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    if not project_path.exists() or not project_path.is_dir():
        print(f"Error: project path does not exist or is not a directory: {project_path}", file=sys.stderr)
        return 1

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_dirs.update(args.exclude_dir)

    module_map_path = (
        Path(args.module_map).resolve()
        if args.module_map
        else (project_path / DEFAULT_MODULE_MAP_FILENAME)
    )
    if args.create_module_map_template:
        created = create_module_map_template(
            module_map_path=module_map_path,
            force=args.force_module_map_template,
        )
        if created:
            print(f"Created module map template at {module_map_path}")
        else:
            print(
                "Module map template already exists. Use --force-module-map-template to overwrite.",
                file=sys.stderr,
            )
            return 1
        return 0

    loaded_module_map = load_module_map(module_map_path) if module_map_path.exists() else {}

    if args.max_depth < -1:
        print("Error: --max-depth must be -1 or a non-negative integer", file=sys.stderr)
        return 1
    max_depth = None if args.max_depth == -1 else args.max_depth

    report = scan_imports(
        project_path=project_path,
        excluded_dirs=excluded_dirs,
        include_relative=args.include_relative,
        max_depth=max_depth,
        show_parse_warnings=args.show_parse_warnings,
    )

    requirements_comparison: RequirementsComparison | None = None
    if args.requirements:
        requirements_path = Path(args.requirements).resolve()
        if not requirements_path.exists() or not requirements_path.is_file():
            print(f"Error: requirements file does not exist: {requirements_path}", file=sys.stderr)
            return 1
        requirements_packages = parse_requirements_file(requirements_path)
        ignored_imports, local_modules = build_ignored_imports(
            project_path=project_path,
            excluded_dirs=excluded_dirs,
            max_depth=max_depth,
            include_stdlib=args.include_stdlib,
            include_local=args.include_local,
            extra_ignored=args.ignore_module,
        )
        requirements_comparison = compare_imports_to_requirements(
            report,
            requirements_packages,
            ignored_imports=ignored_imports,
            local_modules=local_modules,
            module_map=loaded_module_map,
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_stream:
            if args.format == "json":
                write_json(report, output_stream, requirements_comparison=requirements_comparison)
            elif args.format == "csv":
                write_csv(report, output_stream, sort_by=args.sort_by)
            else:
                write_text(
                    report,
                    output_stream,
                    requirements_comparison=requirements_comparison,
                    sort_by=args.sort_by,
                )
        print(f"Wrote report to {output_path}")
        if args.format == "csv" and requirements_comparison is not None:
            print("Note: requirements comparison is not embedded in CSV output.", file=sys.stderr)
        return 0

    if args.format == "json":
        write_json(report, sys.stdout, requirements_comparison=requirements_comparison)
    elif args.format == "csv":
        write_csv(report, sys.stdout, sort_by=args.sort_by)
        if requirements_comparison is not None:
            print("Note: requirements comparison is not embedded in CSV output.", file=sys.stderr)
    else:
        write_text(
            report,
            sys.stdout,
            requirements_comparison=requirements_comparison,
            sort_by=args.sort_by,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
