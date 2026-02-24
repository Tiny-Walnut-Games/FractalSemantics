from __future__ import annotations

import json
from pathlib import Path

import comprehensive_experiment_analysis as cea


def test_extract_numeric_collision_rates_handles_multiple_shapes() -> None:
    assert cea._extract_numeric_collision_rates({"a": 1, "b": "2.5", "c": "x", "d": True}) == [1.0, 2.5]
    assert cea._extract_numeric_collision_rates([0.1, "0.2", None, "x", False]) == [0.1, 0.2]
    assert cea._extract_numeric_collision_rates(0.9) == [0.9]
    assert cea._extract_numeric_collision_rates(None) == []


def test_write_audit_log_jsonl_structure(tmp_path: Path) -> None:
    audit_path = tmp_path / "archive" / "audit.jsonl"

    cea._write_audit_log(
        action="archive_move",
        outcome="success",
        details={"source": "a", "destination": "b"},
        audit_log_path=audit_path,
    )

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["action"] == "archive_move"
    assert entry["outcome"] == "success"
    assert isinstance(entry.get("timestamp"), str)
    assert isinstance(entry.get("actor"), str)
    assert entry["details"] == {"source": "a", "destination": "b"}


def test_archive_paths_continues_when_hash_fails(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True)

    src = results_dir / "example.json"
    src.write_text('{"ok": true}', encoding="utf-8")

    def _raise_hash(_path: Path) -> str:
        raise OSError("hash failed")

    monkeypatch.setattr(cea, "_file_sha256", _raise_hash)

    archive_dir = cea._archive_paths([src], results_dir, project_root)

    assert archive_dir is not None
    archived_file = archive_dir / "example.json"
    assert archived_file.exists()

    manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 1
    assert "sha256" not in manifest["entries"][0]


def test_archive_paths_logs_move_failure(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True)

    src = results_dir / "bad.json"
    src.write_text("{}", encoding="utf-8")

    original_move = cea.shutil.move

    def _failing_move(src_path: str, dst_path: str):
        if src_path.endswith("bad.json"):
            raise OSError("move failed")
        return original_move(src_path, dst_path)

    monkeypatch.setattr(cea.shutil, "move", _failing_move)

    archive_dir = cea._archive_paths([src], results_dir, project_root)

    assert archive_dir is not None
    assert src.exists()

    audit_log = project_root / "results" / "archive" / "audit.jsonl"
    assert audit_log.exists()
    entries = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(e.get("outcome") == "failure" for e in entries)
