import io
from pathlib import Path
from types import SimpleNamespace

from imports_reporter import (
    build_ignored_imports,
    compare_imports_to_requirements,
    create_module_map_template,
    load_module_map,
    parse_requirements_file,
    scan_imports,
    get_stdlib_module_names,
    write_csv,
    write_text,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scan_imports_max_depth_boundary_and_exclusion(tmp_path: Path) -> None:
    _write(tmp_path / "root.py", "import json\n")
    _write(tmp_path / "a" / "b" / "c" / "d" / "e" / "depth5.py", "import requests\n")
    _write(tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "depth6.py", "import numpy\n")

    report = scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
    )

    assert "json" in report
    assert "requests" in report
    assert "numpy" not in report


def test_scan_imports_unlimited_depth_includes_deeper_files(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "deep.py", "import numpy\n")

    report = scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=None,
    )

    assert "numpy" in report


def test_compare_requirements_ignores_stdlib_and_local_by_default(tmp_path: Path) -> None:
    _write(tmp_path / "my_local.py", "")
    _write(
        tmp_path / "main.py",
        "import json\nimport my_local\nimport requests\n",
    )

    report = scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
    )

    ignored, local_modules = build_ignored_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        max_depth=5,
        include_stdlib=False,
        include_local=False,
        extra_ignored=[],
    )
    comparison = compare_imports_to_requirements(
        report=report,
        requirements_packages={"requests"},
        ignored_imports=ignored,
        local_modules=local_modules,
    )

    assert comparison["imported_not_in_requirements"] == []
    assert comparison["requirements_not_imported"] == []
    assert "json" in comparison["ignored_imports"]
    assert "my_local" in comparison["ignored_imports"]
    assert "my_local" in comparison["ignored_local_modules"]


def test_compare_requirements_with_manual_ignore(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", "import yaml\n")

    report = scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
    )

    comparison = compare_imports_to_requirements(
        report=report,
        requirements_packages=set(),
        ignored_imports={"yaml"},
    )

    assert comparison["imported_not_in_requirements"] == []
    assert comparison["requirements_not_imported"] == []
    assert comparison["ignored_imports"] == ["yaml"]


def test_scan_imports_parse_warnings_opt_in(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "broken.py", "def nope(:\n")

    scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
    )
    first_capture = capsys.readouterr()
    assert "syntax error" not in first_capture.err.lower()

    scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
        show_parse_warnings=True,
    )
    second_capture = capsys.readouterr()
    assert "syntax error" in second_capture.err.lower()


def test_recursive_local_module_resolver_ignores_subfolder_modules(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "internal_mod.py", "def value():\n    return 1\n")
    _write(tmp_path / "feature" / "consumer.py", "import internal_mod\nimport requests\n")

    report = scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
    )

    ignored, local_modules = build_ignored_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        max_depth=5,
        include_stdlib=False,
        include_local=False,
        extra_ignored=[],
    )
    comparison = compare_imports_to_requirements(
        report=report,
        requirements_packages={"requests"},
        ignored_imports=ignored,
        local_modules=local_modules,
    )

    assert "internal_mod" in local_modules
    assert "internal_mod" in comparison["ignored_local_modules"]
    assert comparison["imported_not_in_requirements"] == []


def test_module_map_template_create_and_load(tmp_path: Path) -> None:
    module_map_path = tmp_path / ".imports_reporter_module_map.json"

    created = create_module_map_template(module_map_path)
    assert created is True
    assert module_map_path.exists()

    loaded = load_module_map(module_map_path)
    assert loaded["sklearn"] == "scikit_learn"
    assert loaded["yaml"] == "pyyaml"

    created_again = create_module_map_template(module_map_path)
    assert created_again is False


def test_compare_requirements_applies_module_map(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", "import sklearn\n")

    report = scan_imports(
        project_path=tmp_path,
        excluded_dirs=set(),
        include_relative=False,
        max_depth=5,
    )

    comparison = compare_imports_to_requirements(
        report=report,
        requirements_packages={"scikit_learn"},
        ignored_imports=set(),
        module_map={"sklearn": "scikit_learn"},
    )

    assert comparison["imported_not_in_requirements"] == []
    assert comparison["requirements_not_imported"] == []
    assert comparison["applied_module_map"] == {"sklearn": "scikit_learn"}


def test_parse_requirements_file_follows_nested_includes(tmp_path: Path) -> None:
    _write(tmp_path / "requirements.txt", "-r requirements-core.txt\n-r requirements-extra.txt\n")
    _write(tmp_path / "requirements-core.txt", "numpy>=1.20.0\n")
    _write(tmp_path / "requirements-extra.txt", "pandas>=2.0.0\n")

    packages = parse_requirements_file(tmp_path / "requirements.txt")

    assert packages == {"numpy", "pandas"}


def test_write_text_can_sort_by_import_count() -> None:
    report = {
        "beta": {"import_count": 1, "file_count": 1, "files": ["b.py"]},
        "alpha": {"import_count": 3, "file_count": 1, "files": ["a.py"]},
    }

    output = io.StringIO()
    write_text(report, output, sort_by="import_count")
    lines = [line for line in output.getvalue().splitlines() if line.strip()]

    assert lines[0].startswith("alpha ")
    assert lines[1].startswith("beta ")


def test_write_csv_can_sort_by_file_count() -> None:
    report = {
        "beta": {"import_count": 4, "file_count": 1, "files": ["b.py"]},
        "alpha": {"import_count": 1, "file_count": 2, "files": ["a.py", "a2.py"]},
    }

    output = io.StringIO()
    write_csv(report, output, sort_by="file_count")
    rows = [line for line in output.getvalue().splitlines() if line.strip()]

    assert rows[1].startswith("alpha,")
    assert rows[2].startswith("beta,")


def test_get_stdlib_module_names_fallback_works_without_stdlib_module_names() -> None:
    fake_sys = SimpleNamespace(builtin_module_names=("sys", "math"))

    names = get_stdlib_module_names(runtime_sys=fake_sys)

    assert "sys" in names
    assert "math" in names
    assert "builtins" in names
