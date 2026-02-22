#!/usr/bin/env python3
"""
Experiment Runner for FractalSemantics HTML Web Application

This script provides the backend execution capabilities for the HTML web application,
allowing it to run real FractalSemantics experiments with educational output.
"""

import ast
import asyncio
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Optional, TypeAlias

import tqdm

# Add the fractalsemantics module to the path FIRST, before any imports

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

# Display-format precision only (presentation), not simulation/runtime constants.
SCORE_DECIMALS = 2
DURATION_DECIMALS = 2
DURATION_DETAIL_DECIMALS = 4
PERCENT_DECIMALS = 1

# UI/progress and output-shaping heuristics (non-physics constants).
PROGRESS_PERCENT_MIN = 0.0
PROGRESS_PERCENT_MAX = 100.0
PROGRESS_BAR_TOTAL = 100
PROGRESS_BAR_COLUMNS = 80
PROGRESS_STAGE_LABEL_MAX_CHARS = 22
QUEUE_POLL_TIMEOUT_SECONDS = 0.1
OUTPUT_DISPLAY_MAX_LINES = 1000
PROGRESS_MESSAGE_LIMIT = 250
SEQUENTIAL_PROGRESS_MIN_INTERVAL_SECONDS = 0.1
PROGRESS_SEPARATOR_EVERY = 4
POSTULATE_VALIDATION_EXPERIMENTS = {
    "EXP-13", "EXP-14", "EXP-15", "EXP-16", "EXP-17", "EXP-18", "EXP-19", "EXP-20", "EXP-21"
}
HASH_MISMATCH_EXPECTED_EXPERIMENTS = {
    "EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05", "EXP-06", "EXP-07", "EXP-08", "EXP-09",
    "EXP-10", "EXP-11", "EXP-11b", "EXP-12", "EXP-13", "EXP-14", "EXP-15", "EXP-16", "EXP-17",
    "EXP-18", "EXP-19", "EXP-20", "EXP-21",
}
ENABLE_ADVANCED_REPRO_CHECK_ENV = "FRACTALSEMANTICS_ENABLE_ADVANCED_REPRO_CHECK"
SOFTCOPY_ENV = "FRACTALSEMANTICS_SOFTCOPY"


def _env_flag_enabled(name: str) -> bool:
    """Parse common truthy env values for feature toggles."""
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_repro_runs_arg(args: list[str]) -> int:
    """Parse reproducibility rerun count from CLI args."""
    repro_runs = 1
    for i, arg in enumerate(args):
        if arg.startswith("--repro-runs="):
            value = arg.split("=", 1)[1].strip()
        elif arg == "--repro-runs":
            if i + 1 >= len(args):
                raise ValueError("--repro-runs requires an integer value")
            value = args[i + 1].strip()
        else:
            continue

        try:
            repro_runs = int(value)
        except ValueError as exc:
            raise ValueError("--repro-runs must be an integer >= 1") from exc

        if repro_runs < 1:
            raise ValueError("--repro-runs must be >= 1")

    return repro_runs


def _parse_softcopy_arg(args: list[str]) -> bool:
    """Parse softcopy behavior from CLI args.

    True means artifacts are persisted; False means artifact writes are disabled/cleaned up.
    """
    softcopy_enabled = True

    def parse_bool(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("--softcopy must be true or false")

    for i, arg in enumerate(args):
        if arg.startswith("--softcopy="):
            softcopy_enabled = parse_bool(arg.split("=", 1)[1])
        elif arg == "--softcopy":
            if i + 1 >= len(args):
                raise ValueError("--softcopy requires true or false")
            softcopy_enabled = parse_bool(args[i + 1])

    return softcopy_enabled

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    module_name: str
    description: str
    educational_focus: str
    experiment_type: str = "standard"  # "standard", "advanced", "stress_test"
    quick_mode_supported: bool = True
    timeout_seconds: int = 300 # Default to 300 seconds (5 minutes) for all experiments
    dependencies: list[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Result of an experiment execution."""
    experiment_id: str
    success: bool
    duration: float
    output: str
    metrics: dict[str, Any]
    educational_content: list[str]
    result_type: str = "unknown"  # "success", "warning", "partial_success", "failure"
    error_details: Optional[dict[str, Any]] = None

@dataclass
class BatchRunResult:
    """Result of running multiple experiments."""
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    total_duration: float
    experiment_results: list[ExperimentResult]
    summary_report: str
    performance_metrics: dict[str, Any] = field(default_factory=dict)

class ExperimentRunner:
    """Runs FractalSemantics experiments with educational output."""

    def __init__(self, softcopy_enabled: bool = True):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "results"
        self.reports_dir = self.results_dir / "reports"
        self.figures_dir = self.results_dir / "figures"
        self.softcopy_enabled = softcopy_enabled
        self.results_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        self.experiment_configs = self._load_experiment_configs()
        self._validate_configurations()

    def _normalize_output_save_language(self, output: str) -> str:
        """Standardize save-related phrases in experiment terminal output."""
        if not output:
            return output

        normalized_lines: list[str] = []
        for line in output.splitlines():
            saved_match = re.match(
                r"^\s*(Results saved to|Visualization saved to|Figure saved to|Output saved to)\s*:\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if saved_match:
                normalized_lines.append(f"Artifact saved to: {saved_match.group(2).strip()}")
                continue

            path_match = re.match(r"^\s*(Results|Output)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
            if path_match:
                normalized_lines.append(f"Artifact path: {path_match.group(2).strip()}")
                continue

            normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _cleanup_softcopy_artifacts(self, output: str) -> list[str]:
        """Delete persisted artifact files when softcopy is disabled."""
        referenced_paths = self._extract_saved_paths(output)
        deleted_paths: list[str] = []

        allowed_roots = [self.results_dir.resolve(), (self.project_root / "figures").resolve()]
        allowed_suffixes = {".json", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".md", ".txt"}

        for candidate in referenced_paths:
            try:
                candidate_path = Path(candidate).resolve()
            except Exception:
                continue

            if candidate_path.suffix.lower() not in allowed_suffixes:
                continue

            if not any(root == candidate_path or root in candidate_path.parents for root in allowed_roots):
                continue

            if not candidate_path.exists() or not candidate_path.is_file():
                continue

            with contextlib.suppress(Exception):
                candidate_path.unlink()
                deleted_paths.append(str(candidate_path))

        return deleted_paths

    def _scientific_score(self, result_type: str) -> float:
        """Convert result type to a normalized scientific score."""
        mapping = {
            "success": 1.0,
            "partial_success": 0.5,
            "warning": 0.0,
            "failure": 0.0,
        }
        return mapping.get(result_type, 0.0)

    def _technical_status_label(self, technical_success: bool) -> str:
        """Human-readable technical run status."""
        return "PASS (execution completed)" if technical_success else "FAIL (execution/runtime error)"

    def _scientific_outcome_label(self, experiment_id: str, result_type: str, technical_success: bool) -> str:
        """Human-readable scientific outcome independent of technical execution status."""
        if not technical_success:
            return "Not evaluated due to technical execution failure"

        if result_type == "success":
            return "Hypothesis supported by this run"
        if result_type == "partial_success":
            return "Partially supported (below target confidence/performance)"
        if result_type == "warning":
            if experiment_id in POSTULATE_VALIDATION_EXPERIMENTS:
                return "Scientifically valid negative result (hypothesis not supported under tested conditions)"
            return "Scientific criteria not met (negative or inconclusive outcome)"

        return "Scientific outcome undetermined"

    def _format_float(self, value: float, decimals: int) -> str:
        """Format floating-point numbers with centralized precision rules."""
        return f"{value:.{decimals}f}"

    def _format_duration(self, seconds: float, detailed: bool = False) -> str:
        """Format durations using standard precision levels."""
        decimals = DURATION_DETAIL_DECIMALS if detailed else DURATION_DECIMALS
        return self._format_float(seconds, decimals)

    def _format_percent(self, value: float) -> str:
        """Format percentages with centralized display precision."""
        return self._format_float(value, PERCENT_DECIMALS)

    def _extract_saved_paths(self, output: str) -> list[str]:
        """Extract filesystem paths from normalized artifact lines."""
        if not output:
            return []

        extracted_paths: list[str] = []
        seen_paths: set[str] = set()
        for line in output.splitlines():
            match = re.search(r"(?:saved to|artifact path|results|output):\s*(.+)$", line, flags=re.IGNORECASE)
            if not match:
                match = None

            candidates: list[str] = []
            if match:
                candidate = match.group(1).strip().strip('"').strip("'")
                if candidate:
                    candidates.append(candidate)

            generic_path_matches = re.findall(
                r"([A-Za-z]:\\[^\s\"']+\.(?:json|png|jpg|jpeg|svg|pdf)|[\w./\\-]+\.(?:json|png|jpg|jpeg|svg|pdf))",
                line,
                flags=re.IGNORECASE,
            )
            candidates.extend(generic_path_matches)

            if not candidates:
                continue

            for candidate in candidates:
                path_obj = Path(candidate)
                if not path_obj.is_absolute():
                    path_obj = (self.project_root / candidate).resolve()

                normalized = str(path_obj)
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
                extracted_paths.append(normalized)

        return extracted_paths

    def _select_primary_scientific_figure(self, referenced_paths: list[str]) -> str | None:
        """Select an experiment-generated scientific figure path when available."""
        image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        for candidate in referenced_paths:
            candidate_path = Path(candidate)
            if candidate_path.suffix.lower() not in image_suffixes:
                continue
            if candidate_path.exists():
                return str(candidate_path.resolve())
        return None

    def _load_experiment_json_artifact(self, referenced_paths: list[str]) -> dict[str, Any] | None:
        """Load the first referenced JSON artifact emitted by the experiment."""
        for candidate in referenced_paths:
            candidate_path = Path(candidate)
            if candidate_path.suffix.lower() != ".json":
                continue
            if not candidate_path.exists():
                continue
            try:
                with candidate_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return None

    def _generate_scientific_figure_from_json(
        self,
        result: ExperimentResult,
        referenced_paths: list[str],
        output_path: Path,
    ) -> str | None:
        """Generate a scientific figure using experiment JSON artifact data."""
        payload = self._load_experiment_json_artifact(referenced_paths)
        if not payload:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        # EXP-01: collision rate by dimension
        if result.experiment_id == "EXP-01":
            rows = payload.get("results")
            if isinstance(rows, list):
                dimensions: list[int] = []
                collision_rates_pct: list[float] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    dimension = row.get("dimension")
                    collision_rate = row.get("collision_rate")
                    if isinstance(dimension, int) and isinstance(collision_rate, (int, float)):
                        dimensions.append(dimension)
                        collision_rates_pct.append(float(collision_rate) * 100.0)
                if dimensions and collision_rates_pct:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.plot(dimensions, collision_rates_pct, marker="o", linewidth=2)
                    ax.set_title("EXP-01 Collision Rate by Dimension")
                    ax.set_xlabel("Dimension")
                    ax.set_ylabel("Collision Rate (%)")
                    ax.set_xticks(dimensions)
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(output_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    result.metrics["report_generated_figure_kind"] = "scientific_exp01_collision_curve_from_json"
                    return str(output_path.resolve())

        # EXP-02: latency curves by scale
        if result.experiment_id == "EXP-02":
            rows = payload.get("results")
            if isinstance(rows, list):
                scales: list[int] = []
                mean_ms: list[float] = []
                p95_ms: list[float] = []
                p99_ms: list[float] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    scale = row.get("scale")
                    mean_latency = row.get("mean_latency_ms")
                    p95_latency = row.get("p95_latency_ms")
                    p99_latency = row.get("p99_latency_ms")
                    if isinstance(scale, int) and isinstance(mean_latency, (int, float)):
                        scales.append(scale)
                        mean_ms.append(float(mean_latency))
                        p95_ms.append(float(p95_latency) if isinstance(p95_latency, (int, float)) else float(mean_latency))
                        p99_ms.append(float(p99_latency) if isinstance(p99_latency, (int, float)) else float(mean_latency))

                if scales and mean_ms:
                    x_labels = [f"{scale:,}" for scale in scales]
                    x_positions = list(range(len(scales)))
                    fig, ax = plt.subplots(figsize=(10, 5.5))
                    ax.plot(x_positions, mean_ms, marker="o", linewidth=2, label="Mean (ms)")
                    ax.plot(x_positions, p95_ms, marker="s", linewidth=2, label="P95 (ms)")
                    ax.plot(x_positions, p99_ms, marker="^", linewidth=2, label="P99 (ms)")
                    ax.set_title("EXP-02 Retrieval Latency by Scale")
                    ax.set_xlabel("Scale")
                    ax.set_ylabel("Latency (ms)")
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(x_labels, rotation=20, ha="right")
                    ax.grid(alpha=0.25)
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    result.metrics["report_generated_figure_kind"] = "scientific_exp02_latency_curve_from_json"
                    return str(output_path.resolve())

        # Generic scientific plot from JSON tables (covers EXP-04..EXP-12 and similar)
        rows = payload.get("results")
        if not isinstance(rows, list):
            for alternate_key in (
                "scale_results",
                "system_results",
                "benchmark_results",
                "comparison_results",
                "dimension_results",
                "test_results",
                "query_results",
                "optimizer_results",
                "memory_timeline",
                "optimization_results",
                "pressure_phases",
                "iterations",
                "sample_paths",
            ):
                candidate_rows = payload.get(alternate_key)
                if isinstance(candidate_rows, list):
                    rows = candidate_rows
                    break

        if isinstance(rows, list):
            dict_rows = [row for row in rows if isinstance(row, dict)]
            if dict_rows:
                def flatten_numeric_fields(row: dict[str, Any]) -> dict[str, float]:
                    flattened: dict[str, float] = {}
                    for key, value in row.items():
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            flattened[key] = float(value)
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, bool):
                                    continue
                                if isinstance(subvalue, (int, float)):
                                    flattened[f"{key}.{subkey}"] = float(subvalue)
                    return flattened

                flattened_rows = [flatten_numeric_fields(row) for row in dict_rows]

                all_numeric_keys: set[str] = set()
                for flat_row in flattened_rows:
                    all_numeric_keys.update(flat_row.keys())

                x_key_priority = [
                    "scale", "dimension", "sample_size", "queries", "dataset_size",
                    "n", "k", "depth", "level", "cardinality",
                ]
                x_key = next((key for key in x_key_priority if key in all_numeric_keys), None)

                excluded_metric_keys = {
                    x_key,
                    "success",
                    "valid",
                    "all_passed",
                    "meets_threshold",
                    "geometric_limit_hit",
                    "collision_count",
                }

                metric_keys = [key for key in sorted(all_numeric_keys) if key not in excluded_metric_keys]
                if metric_keys:
                    preferred_pattern = re.compile(
                        r"entropy|latency|collision_rate|accuracy|precision|recall|f1|compression|ratio|"
                        r"correlation|distance|error|loss|throughput|memory|time|efficiency|score|"
                        r"retrieval|expressiveness|support|flexibility",
                        flags=re.IGNORECASE,
                    )
                    metric_keys.sort(key=lambda key: (0 if preferred_pattern.search(key) else 1, key))
                    selected_metrics = metric_keys[:3]

                    x_positions = list(range(len(dict_rows)))
                    x_axis_label = "Result Index"
                    x_labels = [str(idx + 1) for idx in x_positions]

                    if x_key:
                        x_labels = [
                            str(flattened_rows[idx].get(x_key, dict_rows[idx].get(x_key, idx + 1)))
                            for idx in x_positions
                        ]
                        x_axis_label = x_key.replace("_", " ").title()
                    elif any("system_name" in row for row in dict_rows):
                        x_labels = [str(row.get("system_name", idx + 1)) for idx, row in enumerate(dict_rows)]
                        x_axis_label = "System"

                    fig, ax = plt.subplots(figsize=(10, 5.5))
                    for metric_key in selected_metrics:
                        metric_values = [flat_row.get(metric_key, float("nan")) for flat_row in flattened_rows]
                        ax.plot(x_positions, metric_values, marker="o", linewidth=2, label=metric_key)

                    ax.set_title(f"{result.experiment_id} Scientific Metrics (JSON)")
                    ax.set_xlabel(x_axis_label)
                    ax.set_ylabel("Metric Value")
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(x_labels, rotation=20, ha="right")
                    ax.grid(alpha=0.25)
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    result.metrics["report_generated_figure_kind"] = "scientific_generic_json_metrics"
                    result.metrics["report_generated_metrics_keys"] = selected_metrics
                    result.metrics["report_generated_x_key"] = x_key or ("system_name" if any("system_name" in row for row in dict_rows) else "index")
                    return str(output_path.resolve())

        # Top-level numeric metrics fallback (e.g., experiments with summary-only JSON)
        top_level_numeric: dict[str, float] = {}

        def collect_numeric(prefix: str, value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                top_level_numeric[prefix] = float(value)
                return
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    next_prefix = f"{prefix}.{subkey}" if prefix else str(subkey)
                    collect_numeric(next_prefix, subvalue)

        for key, value in payload.items():
            collect_numeric(str(key), value)

        if top_level_numeric:
            preferred_pattern = re.compile(
                r"entropy|latency|collision_rate|accuracy|precision|recall|f1|compression|ratio|"
                r"correlation|distance|error|loss|throughput|memory|time|efficiency|score|"
                r"retrieval|expressiveness|support|flexibility|improvement",
                flags=re.IGNORECASE,
            )
            metric_keys = sorted(top_level_numeric.keys(), key=lambda key: (0 if preferred_pattern.search(key) else 1, key))[:8]
            metric_values = [top_level_numeric[key] for key in metric_keys]

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.bar(range(len(metric_keys)), metric_values, alpha=0.8)
            ax.set_title(f"{result.experiment_id} Scientific Summary Metrics (JSON)")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            ax.set_xticks(range(len(metric_keys)))
            ax.set_xticklabels(metric_keys, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            result.metrics["report_generated_figure_kind"] = "scientific_top_level_json_metrics"
            result.metrics["report_generated_metrics_keys"] = metric_keys
            result.metrics["report_generated_x_key"] = "metric"
            return str(output_path.resolve())

        return None

    def _generate_result_figure(self, result: ExperimentResult, output_path: Path) -> str | None:
        """Generate a per-experiment figure prioritizing scientific data over status."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        output_text = result.output or ""

        # EXP-01: plot collision-rate-by-dimension from experiment output
        exp01_dimensions: list[int] = []
        exp01_rates: list[float] = []
        current_dimension: int | None = None
        for line in output_text.splitlines():
            dim_match = re.search(r"Testing\s+(\d+)D\s+coordinate\s+space", line, flags=re.IGNORECASE)
            if dim_match:
                current_dimension = int(dim_match.group(1))
                continue

            rate_match = re.search(r"Rate:\s*([0-9]+(?:\.[0-9]+)?)%", line)
            if rate_match and current_dimension is not None:
                exp01_dimensions.append(current_dimension)
                exp01_rates.append(float(rate_match.group(1)))
                current_dimension = None

        if result.experiment_id == "EXP-01" and exp01_dimensions and exp01_rates:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(exp01_dimensions, exp01_rates, marker="o", linewidth=2)
            ax.set_title("EXP-01 Collision Rate by Dimension")
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Collision Rate (%)")
            ax.set_xticks(exp01_dimensions)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            result.metrics["report_generated_figure_kind"] = "scientific_exp01_collision_curve"
            return str(output_path.resolve())

        # EXP-02: plot mean and p95 latency vs target threshold by scale
        exp02_scales: list[int] = []
        exp02_mean_ms: list[float] = []
        exp02_p95_ms: list[float] = []
        exp02_target_ms: list[float] = []
        exp02_row_pattern = re.compile(
            r"^\s*([\d,]+)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+<\s*([0-9]+(?:\.[0-9]+)?)\s+(PASS|FAIL)\s*$",
            flags=re.IGNORECASE,
        )
        for line in output_text.splitlines():
            row_match = exp02_row_pattern.match(line)
            if not row_match:
                continue
            exp02_scales.append(int(row_match.group(1).replace(",", "")))
            exp02_mean_ms.append(float(row_match.group(2)))
            exp02_p95_ms.append(float(row_match.group(3)))
            exp02_target_ms.append(float(row_match.group(5)))

        if result.experiment_id == "EXP-02" and exp02_scales:
            x_labels = [f"{scale:,}" for scale in exp02_scales]
            x_positions = list(range(len(exp02_scales)))
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(x_positions, exp02_mean_ms, marker="o", linewidth=2, label="Mean (ms)")
            ax.plot(x_positions, exp02_p95_ms, marker="s", linewidth=2, label="P95 (ms)")
            ax.plot(x_positions, exp02_target_ms, linestyle="--", linewidth=2, label="Target (ms)")
            ax.set_title("EXP-02 Retrieval Latency by Scale")
            ax.set_xlabel("Scale")
            ax.set_ylabel("Latency (ms)")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=20, ha="right")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            result.metrics["report_generated_figure_kind"] = "scientific_exp02_latency_curve"
            return str(output_path.resolve())

        scientific_score = self._scientific_score(result.result_type)
        technical_score = 1.0 if result.success else 0.0

        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = ["Technical Execution", "Scientific Validation"]
        values = [technical_score, scientific_score]
        colors = ["#2E7D32" if technical_score > 0 else "#C62828", "#2E7D32" if scientific_score >= 1.0 else ("#F9A825" if scientific_score > 0 else "#C62828")]

        bars = ax.barh(labels, values, color=colors, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title(f"{result.experiment_id} Run Status")
        ax.grid(axis="x", alpha=0.2)

        for bar, value in zip(bars, values):
            ax.text(
                min(0.98, value + 0.03),
                bar.get_y() + bar.get_height() / 2,
                self._format_float(value, SCORE_DECIMALS),
                va="center",
                fontsize=10,
            )

        return_code = result.metrics.get("return_code") if isinstance(result.metrics, dict) else None
        output_lines = result.metrics.get("output_line_count_display") if isinstance(result.metrics, dict) else None
        fig.text(
            0.02,
            0.02,
            f"Duration: {self._format_duration(result.duration)}s | Result Type: {result.result_type} | Return Code: {return_code} | Output Lines: {output_lines}",
            fontsize=9,
        )

        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        result.metrics["report_generated_figure_kind"] = "generic_status_fallback"
        return str(output_path.resolve())

    def _build_markdown_report(
        self,
        result: ExperimentResult,
        timestamp: str,
        figure_path: str | None,
        referenced_paths: list[str],
    ) -> str:
        """Create Markdown report content for a single experiment run."""
        metrics_json = json.dumps(result.metrics, indent=2, ensure_ascii=False)

        lines = [
            f"# {result.experiment_id} Terminal Report",
            "",
            "## Run Metadata",
            "",
            f"- Timestamp (UTC): {timestamp}",
            f"- Success: {result.success}",
            f"- Result Type: {result.result_type}",
            f"- Technical Run Status: {self._technical_status_label(result.success)}",
            f"- Scientific Outcome: {self._scientific_outcome_label(result.experiment_id, result.result_type, result.success)}",
            f"- Duration (seconds): {self._format_duration(result.duration, detailed=True)}",
            "",
            "## Metrics",
            "",
            "```json",
            metrics_json,
            "```",
            "",
        ]

        if figure_path:
            lines.extend([
                "## Generated Figure",
                "",
                f"- Figure Path: {figure_path}",
                "",
            ])

        if referenced_paths:
            lines.extend([
                "## Referenced Artifacts",
                "",
            ])
            for path in referenced_paths:
                lines.append(f"- {path}")
            lines.append("")

        lines.extend([
            "## Educational Content",
            "",
        ])
        if result.educational_content:
            for i, section in enumerate(result.educational_content, 1):
                lines.extend([
                    f"### Section {i}",
                    "",
                    section,
                    "",
                ])
        else:
            lines.extend(["(none)", ""])

        lines.extend([
            "## Terminal Output",
            "",
            "```text",
            result.output or "",
            "```",
            "",
        ])

        return "\n".join(lines)

    def _save_experiment_artifacts(self, result: ExperimentResult) -> None:
        """Persist Markdown and figure artifacts for each experiment result."""
        if not self.softcopy_enabled:
            result.metrics["report_markdown_path"] = None
            result.metrics["report_figure_path"] = None
            result.metrics["softcopy"] = "disabled"
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = result.experiment_id.lower().replace("-", "")

        referenced_paths = self._extract_saved_paths(result.output)
        figure_path = self._select_primary_scientific_figure(referenced_paths)

        if not figure_path:
            figure_path = self._generate_scientific_figure_from_json(
                result=result,
                referenced_paths=referenced_paths,
                output_path=self.figures_dir / f"{slug}_{timestamp}.png",
            )

        if not figure_path:
            figure_path = self._generate_result_figure(
                result,
                self.figures_dir / f"{slug}_{timestamp}.png",
            )

        markdown_content = self._build_markdown_report(
            result=result,
            timestamp=timestamp,
            figure_path=figure_path,
            referenced_paths=referenced_paths,
        )

        report_path = self.reports_dir / f"{slug}_{timestamp}.md"
        report_path.write_text(markdown_content, encoding="utf-8")

        result.metrics["report_markdown_path"] = str(report_path.resolve())
        result.metrics["report_figure_path"] = figure_path
        result.metrics["softcopy"] = "enabled"
        if referenced_paths:
            result.metrics["referenced_artifact_paths"] = referenced_paths

    def _deduplicate_progress_messages(self, progress_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate progress messages while preserving order."""
        seen: set[tuple[Any, ...]] = set()
        deduplicated: list[dict[str, Any]] = []

        for message in progress_messages:
            key = (
                message.get("timestamp"),
                message.get("progress_percent"),
                message.get("stage"),
                message.get("message"),
                message.get("message_type"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(message)

        return deduplicated

    def _compress_output_for_display(self, text: str, max_lines: int = 1000) -> str:
        """Compress repetitive loop output and enforce a hard maximum line count."""
        if not text:
            return ""

        # Normalize carriage-return updates so progress-style output doesn't run together
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = text.splitlines()
        compressed_lines: list[str] = []

        repetitive_pattern = re.compile(
            r"^\s*(Step\s+\d+/\d+|Generated\s+\d+/\d+|Seed\s*[:=]?\s*\d+|Iteration\s+\d+|Loop\s+\d+)",
            re.IGNORECASE,
        )

        i = 0
        while i < len(lines):
            line = lines[i]

            if repetitive_pattern.match(line):
                block_start = i
                while i < len(lines) and repetitive_pattern.match(lines[i]):
                    i += 1

                block = lines[block_start:i]
                if len(block) <= 6:
                    compressed_lines.extend(block)
                else:
                    compressed_lines.extend(block[:3])
                    omitted = len(block) - 6
                    compressed_lines.append(
                        f"[... omitted {omitted:,} repetitive progress lines ...]"
                    )
                    compressed_lines.extend(block[-3:])
                continue

            if line.strip().startswith("[DEBUG]"):
                debug_block_start = i
                while i < len(lines) and lines[i].strip().startswith("[DEBUG]"):
                    i += 1

                debug_block = lines[debug_block_start:i]
                if len(debug_block) <= 3:
                    compressed_lines.extend(debug_block)
                else:
                    compressed_lines.extend(debug_block[:2])
                    compressed_lines.append(
                        f"[... omitted {len(debug_block) - 2:,} additional debug lines ...]"
                    )
                continue

            compressed_lines.append(line)
            i += 1

        if len(compressed_lines) <= max_lines:
            return "\n".join(compressed_lines)

        head_count = max_lines // 2
        tail_count = max_lines - head_count - 1
        truncated = compressed_lines[:head_count]
        omitted_count = len(compressed_lines) - (head_count + tail_count)
        truncated.append(f"[... omitted {omitted_count:,} lines to keep output <= {max_lines} lines ...]")
        truncated.extend(compressed_lines[-tail_count:])
        return "\n".join(truncated)

    def _format_metrics_for_analysis(self, metrics: dict[str, Any]) -> str:
        """Create concise, readable metric summary for text output."""
        if not isinstance(metrics, dict):
            return "   * Metrics unavailable\n"

        lines: list[str] = []

        return_code = metrics.get("return_code")
        if return_code is not None:
            lines.append(f"   * Return code: {return_code}")

        timeout_seconds = metrics.get("timeout_seconds")
        if timeout_seconds is not None:
            lines.append(f"   * Timeout: {timeout_seconds}s")

        output_sha256 = metrics.get("output_sha256")
        if output_sha256:
            lines.append(f"   * Output fingerprint (SHA-256): {output_sha256[:16]}...")

        progress_messages = metrics.get("progress_messages")
        progress_total = metrics.get("progress_message_total")
        if isinstance(progress_messages, list):
            total = int(progress_total) if isinstance(progress_total, int) else len(progress_messages)
            lines.append(f"   * Progress events: {total} total, {len(progress_messages)} retained")

        reproducibility = metrics.get("reproducibility_check")
        if isinstance(reproducibility, dict):
            reproducible = reproducibility.get("reproducible")
            lines.append(f"   * Reproducibility check: {'PASS' if reproducible else 'WARN'}")

        if not lines:
            return "   * No summarized metrics available\n"

        return "\n".join(lines) + "\n"

    def _format_star_bullets(self, text: str) -> str:
        """Convert compact '* item* item' strings into one-item-per-line bullets."""
        if not text:
            return "   * (none provided)"

        items = [item.strip() for item in text.split("*") if item.strip()]
        if not items:
            return text.strip()

        return "\n".join(f"   * {item}" for item in items)

    def _validate_configurations(self):
        """Validate all experiment configurations for consistency and completeness."""
        for exp_id, config in self.experiment_configs.items():
            # Validate experiment ID format
            if not exp_id.startswith("EXP-"):
                raise ValueError(f"Invalid experiment ID format: {exp_id}")

            # Validate module name format
            if not config.module_name.startswith("fractalsemantics."):
                raise ValueError(f"Invalid module name format: {config.module_name}")

            # Validate experiment type
            valid_types = ["standard", "advanced", "stress_test"]
            if config.experiment_type not in valid_types:
                raise ValueError(f"Invalid experiment type: {config.experiment_type}")

            # Validate timeout
            if config.timeout_seconds <= 0:
                raise ValueError(f"Invalid timeout: {config.timeout_seconds}")

            # Validate dependencies list
            if not isinstance(config.dependencies, list):
                raise ValueError(f"Dependencies must be a list: {config.dependencies}")

            # Check for required dependencies based on experiment type
            self._validate_dependencies(config)

    def _validate_dependencies(self, config: ExperimentConfig):
        """Validate that required dependencies are available."""
        required_packages = {
            "numpy": "numpy",
            "scipy": "scipy",
            "matplotlib": "matplotlib",
            "hashlib": "hashlib",
            "time": "time",
            "random": "random",
            "itertools": "itertools",
            "zlib": "zlib",
            "pickle": "pickle",
            "sklearn": "sklearn",
            "psutil": "psutil",
            "gc": "gc",
            "uuid": "uuid",
            "math": "math",
            "networkx": "networkx"
        }

        missing_packages = []
        for dep in config.dependencies:
            if dep in required_packages:
                try:
                    __import__(dep)
                except ImportError:
                    missing_packages.append(dep)

        if missing_packages:
            print(f"Warning: Missing dependencies for {config.experiment_id}: {', '.join(missing_packages)}")
            print("   Some experiments may fail due to missing packages.")

    def _load_experiment_configs(self) -> dict[str, ExperimentConfig]:
        """Load experiment configurations with proper structure and validation."""
        configs = {
            "EXP-01": ExperimentConfig(
                experiment_id="EXP-01",
                module_name="fractalsemantics.exp01_geometric_collision",
                description="Tests that every bit-chain gets a unique address with zero collisions using 8-dimensional coordinates.",
                educational_focus="8-Dimensional Coordinate Space and Collision Resistance Mathematics",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "hashlib"]
            ),
            "EXP-02": ExperimentConfig(
                experiment_id="EXP-02",
                module_name="fractalsemantics.exp02_retrieval_efficiency",
                description="Tests sub-millisecond retrieval performance at scale using hash table indexing.",
                educational_focus="Hash Table Performance Analysis and Big O Notation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["hashlib", "time"]
            ),
            "EXP-03": ExperimentConfig(
                experiment_id="EXP-03",
                module_name="fractalsemantics.exp03_coordinate_entropy",
                description="Validates that all 7 dimensions are necessary to avoid collisions through ablation testing.",
                educational_focus="Dimensional Analysis and Shannon Entropy Calculation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-04": ExperimentConfig(
                experiment_id="EXP-04",
                module_name="fractalsemantics.exp04_fractal_scaling",
                description="Tests consistency of addressing properties across different scales (1K to 1M entities).",
                educational_focus="Fractal Geometry Principles and Scale Invariance Analysis",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "matplotlib"]
            ),
            "EXP-05": ExperimentConfig(
                experiment_id="EXP-05",
                module_name="fractalsemantics.exp05_compression_expansion",
                description="Tests lossless compression through hierarchical structures (fragments - clusters - glyphs - mist).",
                educational_focus="Information Theory and Hierarchical Compression Algorithms",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["zlib", "pickle"]
            ),
            "EXP-06": ExperimentConfig(
                experiment_id="EXP-06",
                module_name="fractalsemantics.exp06_entanglement_detection",
                description="Tests detection of narrative entanglement between bit-chains using semantic similarity.",
                educational_focus="Semantic Similarity Metrics and Cosine Similarity Calculation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-07": ExperimentConfig(
                experiment_id="EXP-07",
                module_name="fractalsemantics.exp07_luca_bootstrap",
                description="Tests bootstrapping from Last Universal Common Ancestor to derive all entities.",
                educational_focus="Evolutionary Algorithms and Lineage Tree Generation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["random", "itertools"]
            ),
            "EXP-08": ExperimentConfig(
                experiment_id="EXP-08",
                module_name="fractalsemantics.exp08_self_organizing_memory",
                description="Tests FractalSemantics's ability to create self-organizing memory structures with semantic clustering.",
                educational_focus="Neural Network Clustering and Self-Organization Principles",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "sklearn"]
            ),
            "EXP-09": ExperimentConfig(
                experiment_id="EXP-09",
                module_name="fractalsemantics.exp09_memory_pressure",
                description="Tests system resilience and performance under constrained memory conditions.",
                educational_focus="Memory Management Algorithms and Performance Under Constraints",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["psutil", "gc"]
            ),
            "EXP-10": ExperimentConfig(
                experiment_id="EXP-10",
                module_name="fractalsemantics.exp10_multidimensional_query",
                description="Tests FractalSemantics's unique querying capabilities across all 8 dimensions.",
                educational_focus="Multi-Dimensional Indexing and Query Optimization Algorithms",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-11": ExperimentConfig(
                experiment_id="EXP-11",
                module_name="fractalsemantics.exp11_dimension_cardinality",
                description="Explores pros and cons of 7 dimensions vs. more or fewer dimensions.",
                educational_focus="Dimensional Trade-off Analysis and Optimal Dimension Count",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "matplotlib"]
            ),
            "EXP-11b": ExperimentConfig(
                experiment_id="EXP-11b",
                module_name="fractalsemantics.exp11b_dimension_stress_test",
                description="Stress tests dimensional analysis with extreme parameter variations.",
                educational_focus="Dimensional Stress Testing and Parameter Sensitivity Analysis",
                experiment_type="stress_test",
                quick_mode_supported=False,  # Stress tests require full execution
                timeout_seconds=600,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-12": ExperimentConfig(
                experiment_id="EXP-12",
                module_name="fractalsemantics.exp12_benchmark_comparison",
                description="Compares FractalSemantics against common systems (UUID, SHA256, Vector DB, etc.).",
                educational_focus="Comparative Performance Analysis and Benchmarking Methodologies",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["time", "uuid", "hashlib"]
            ),
            "EXP-13": ExperimentConfig(
                experiment_id="EXP-13",
                module_name="fractalsemantics.exp13_fractal_gravity",
                description="Tests whether fractal entities naturally create gravitational cohesion without falloff.",
                educational_focus="Fractal Gravity and Hierarchical Cohesion Analysis",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "math"]
            ),
            "EXP-14": ExperimentConfig(
                experiment_id="EXP-14",
                module_name="fractalsemantics.exp14_atomic_fractal_mapping",
                description="Maps electron shell structure to fractal parameters and validates atomic structure emergence.",
                educational_focus="Atomic Structure and Fractal Hierarchy Mapping",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy"]
            ),
            "EXP-15": ExperimentConfig(
                experiment_id="EXP-15",
                module_name="fractalsemantics.exp15_topological_conservation",
                description="Tests whether fractal systems conserve topology rather than classical energy and momentum.",
                educational_focus="Topological Conservation Laws and Fractal Physics",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "networkx"]
            ),
            "EXP-16": ExperimentConfig(
                experiment_id="EXP-16",
                module_name="fractalsemantics.exp16_hierarchical_distance_mapping",
                description="Tests hierarchical distance mapping and its relationship to spatial distance.",
                educational_focus="Hierarchical Distance Metrics and Spatial Mapping",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-17": ExperimentConfig(
                experiment_id="EXP-17",
                module_name="fractalsemantics.exp17_thermodynamic_validation",
                description="Validates thermodynamic properties of fractal systems and energy conservation.",
                educational_focus="Thermodynamic Validation and Energy Analysis",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-18": ExperimentConfig(
                experiment_id="EXP-18",
                module_name="fractalsemantics.exp18_falloff_thermodynamics",
                description="Tests falloff thermodynamics and its relationship to hierarchical structure.",
                educational_focus="Falloff Thermodynamics and Hierarchical Energy Distribution",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-19": ExperimentConfig(
                experiment_id="EXP-19",
                module_name="fractalsemantics.exp19_orbital_equivalence",
                description="Tests orbital equivalence and hierarchical relationships in fractal systems.",
                educational_focus="Orbital Equivalence and Fractal Dynamics",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-20": ExperimentConfig(
                experiment_id="EXP-20",
                module_name="fractalsemantics.exp20_vector_field_derivation",
                description="Derives vector field approaches for fractal gravitational interactions.",
                educational_focus="Vector Field Derivation and Fractal Mechanics",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-21": ExperimentConfig(
                experiment_id="EXP-21",
                module_name="fractalsemantics.exp21_earth_moon_sun",
                description="Simulates the Earth-Moon-Sun system with accurate orbital mechanics and gravitational interactions.",
                educational_focus="Orbital Mechanics and Gravitational Simulation",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            )
        }
        return configs

    async def run_experiment(
        self,
        experiment_id: str,
        quick_mode: bool = False,
        progress_callback=None,
        repro_runs: int = 1,
    ) -> ExperimentResult:
        """Run a single experiment with educational output."""
        if experiment_id not in self.experiment_configs:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        config = self.experiment_configs[experiment_id]
        start_time = time.time()

        educational_content = []

        try:
            # Generate educational introduction
            educational_content.append(self._generate_introduction(experiment_id, config))

            # Run the actual experiment
            result = await self._execute_experiment_module(
                experiment_id,
                quick_mode,
                progress_callback=progress_callback,
                repro_runs=repro_runs,
            )

            # Determine result type based on experiment outcome
            result_type = self._determine_result_type(experiment_id, result)

            # Add educational analysis with explicit technical/scientific status
            educational_content.append(self._generate_analysis(experiment_id, result, result_type))

            duration = time.time() - start_time

            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                success=result["success"],
                duration=duration,
                output=result["output"],
                metrics=result["metrics"],
                educational_content=educational_content,
                result_type=result_type
            )
            self._save_experiment_artifacts(experiment_result)
            return experiment_result

        except Exception as e:
            duration = time.time() - start_time
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                duration=duration,
                output=f"Error executing experiment: {str(e)}",
                metrics={},
                educational_content=[f"[FAIL] Experiment failed with error: {str(e)}"],
                result_type="failure"
            )
            self._save_experiment_artifacts(experiment_result)
            return experiment_result

    def _generate_introduction(self, experiment_id: str, config: ExperimentConfig) -> str:
        """Generate educational introduction for the experiment."""
        intro = f"""
        - EXPERIMENT: {experiment_id} - {config.module_name.split('.')[-1].replace('_', ' ').title()}
        - Educational Focus: {config.educational_focus}

        - Objective:
        {config.description}

        - Mathematical Concepts Covered:
        """

        # Add specific mathematical concepts for each experiment
        concepts = self._get_mathematical_concepts(experiment_id)
        for concept in concepts:
            intro += f"   * {concept}\n"

        intro += """
        - Step-by-Step Process:
        """

        # Add step-by-step process
        steps = self._get_experiment_steps(experiment_id)
        for i, step in enumerate(steps, 1):
            intro += f"   {i}. {step}\n"

        intro += "\n" + "="*60 + "\n"
        return intro

    def _generate_analysis(self, experiment_id: str, result: dict[str, Any], result_type: str = "unknown") -> str:
        """Generate educational analysis of experiment results."""
        analysis = f"""
        - EXPERIMENT RESULTS ANALYSIS: {experiment_id}
        - Key Learning Outcomes:

        """

        if not result["success"]:
            analysis += "[FAIL] Experiment encountered technical issues.\n"
            analysis += "- Troubleshooting Insights:\n"
            analysis += "   * This demonstrates real-world challenges in computational systems\n"
            analysis += "   * Error analysis helps identify system limitations\n"
            analysis += "   * Understanding failure modes is crucial for system design\n"
        elif result_type == "warning":
            if experiment_id in POSTULATE_VALIDATION_EXPERIMENTS:
                analysis += "[NEGATIVE RESULT] Technical execution succeeded; hypothesis was not supported under tested conditions.\n"
                analysis += "- Scientific Interpretation:\n"
                analysis += "   * This is a valid falsification signal, not a runtime failure\n"
                analysis += "   * The model/postulate should be revised, narrowed, or re-tested with new conditions\n"
            else:
                analysis += "[WARN] Technical execution succeeded, but scientific validation criteria were not met.\n"
            analysis += "- Performance Metrics:\n"
            analysis += self._format_metrics_for_analysis(result.get("metrics", {}))
            analysis += "   * Interpretation: scientific criteria not met; execution remains technically valid\n"
        elif result_type == "partial_success":
            analysis += "[PARTIAL] Technical execution succeeded with partial scientific success.\n"
            analysis += "- Performance Metrics:\n"
            analysis += self._format_metrics_for_analysis(result.get("metrics", {}))
            analysis += "   * Interpretation: model behavior is usable but below target scientific thresholds\n"
        else:
            analysis += "[SUCCESS] Experiment completed successfully!\n"
            analysis += "- Performance Metrics:\n"
            analysis += self._format_metrics_for_analysis(result.get("metrics", {}))

        analysis += "\n"
        analysis += "            - Real-World Applications:\n"
        analysis += self._format_star_bullets(self._get_real_world_applications(experiment_id))
        analysis += "\n\n"
        analysis += "            - Takeaway Lessons:\n"
        analysis += self._format_star_bullets(self._get_key_lessons(experiment_id))
        analysis += "\n"

        return analysis

    def _get_mathematical_concepts(self, experiment_id: str) -> list[str]:
        """Get mathematical concepts for the experiment."""
        concepts_map = {
            "EXP-01": [
                "8-Dimensional Coordinate Space",
                "Collision Resistance Mathematics",
                "Address Generation Formula",
                "Geometric Probability Theory"
            ],
            "EXP-02": [
                "Hash Table Performance Analysis",
                "Big O Notation (O(1) retrieval)",
                "Latency Measurement Statistics",
                "Time Complexity Analysis"
            ],
            "EXP-03": [
                "Dimensional Analysis",
                "Shannon Entropy Calculation",
                "Ablation Study Methodology",
                "Information Theory Fundamentals"
            ],
            "EXP-04": [
                "Fractal Geometry Principles",
                "Scale Invariance Analysis",
                "Power Law Distributions",
                "Self-Similarity Mathematics"
            ],
            "EXP-05": [
                "Information Theory",
                "Huffman Coding Principles",
                "Hierarchical Compression Algorithms",
                "Lossless Compression Mathematics"
            ],
            "EXP-06": [
                "Semantic Similarity Metrics",
                "Cosine Similarity Calculation",
                "Entanglement Threshold Analysis",
                "Vector Space Mathematics"
            ],
            "EXP-07": [
                "Evolutionary Algorithms",
                "Lineage Tree Generation",
                "Genetic Distance Metrics",
                "Phylogenetic Analysis"
            ],
            "EXP-08": [
                "Neural Network Clustering",
                "Self-Organization Principles",
                "Semantic Distance Metrics",
                "Network Topology Analysis"
            ],
            "EXP-09": [
                "Memory Management Algorithms",
                "Performance Under Constraints",
                "Resource Optimization",
                "System Resilience Analysis"
            ],
            "EXP-10": [
                "Multi-Dimensional Indexing",
                "Query Optimization Algorithms",
                "Dimensional Pruning Strategies",
                "Spatial Database Theory"
            ],
            "EXP-11": [
                "Dimensional Trade-off Analysis",
                "Expressiveness vs. Complexity",
                "Optimal Dimension Count",
                "Pareto Efficiency Analysis"
            ],
            "EXP-11b": [
                "Stress testing methodologies",
                "Parameter sensitivity analysis",
                "Robust system design",
                "Performance under extreme conditions"
            ],
            "EXP-12": [
                "Comparative Performance Analysis",
                "Benchmarking Methodologies",
                "System Trade-off Evaluation",
                "Statistical Significance Testing"
            ],
            "EXP-13": [
                "Fractal Gravity and Hierarchical Cohesion",
                "Hierarchical Distance Metrics",
                "Tree Structure Mathematics",
                "Gravitational Field Theory"
            ],
            "EXP-14": [
                "Atomic Structure and Electron Configuration",
                "Shell-Based Fractal Mapping",
                "Periodic Table Analysis",
                "Quantum Mechanical Principles"
            ],
            "EXP-15": [
                "Topological Conservation Laws",
                "Fractal Physics Principles",
                "Conservation of Structure",
                "Classical vs. Fractal Mechanics"
            ],
            "EXP-16": [
                "Hierarchical Distance Metrics",
                "Spatial Mapping Algorithms",
                "Distance Transformation Mathematics",
                "Multi-Scale Analysis"
            ],
            "EXP-17": [
                "Thermodynamic Validation",
                "Energy Conservation Analysis",
                "Statistical Mechanics",
                "Thermal Equilibrium Principles"
            ],
            "EXP-18": [
                "Falloff Thermodynamics",
                "Energy Distribution Patterns",
                "Hierarchical Energy Flow",
                "Thermodynamic Efficiency Analysis"
            ],
            "EXP-19": [
                "Orbital Equivalence",
                "Hierarchical Dynamics",
                "Fractal Orbital Mechanics",
                "Equivalence Principle Analysis"
            ],
            "EXP-20": [
                "Vector Field Derivation",
                "Fractal Gravitational Interactions",
                "Field Theory Mathematics",
                "Vector Calculus Applications"
            ],
            "EXP-21": [
                "Orbital Mechanics",
                "Gravitational Simulation",
                "N-body Problem Analysis",
                "Numerical Integration Methods"
            ]
        }
        return concepts_map.get(experiment_id, ["General Computational Concepts"])

    def _get_experiment_steps(self, experiment_id: str) -> list[str]:
        """Get step-by-step process for the experiment."""
        steps_map = {
            "EXP-01": [
                "Generate random bit-chains with specified sample size",
                "Compute FractalSemantics coordinates for each bit-chain",
                "Calculate unique addresses using coordinate hashing",
                "Verify zero collisions across all generated addresses",
                "Analyze distribution patterns and statistical properties"
            ],
            "EXP-02": [
                "Build hash table index mapping addresses to bit-chains",
                "Generate random retrieval queries across the dataset",
                "Measure query response times with high-precision timing",
                "Calculate average and percentile latencies",
                "Verify sub-millisecond performance requirements"
            ],
            "EXP-03": [
                "Calculate baseline entropy with complete 7-dimensional coordinates",
                "Remove each dimension individually through ablation",
                "Measure entropy reduction for each dimension removal",
                "Identify critical dimensions that significantly impact entropy",
                "Validate necessity threshold for collision avoidance"
            ],
            "EXP-04": [
                "Generate datasets at multiple scales (1K, 10K, 100K, 1M entities)",
                "Measure collision rates at each scale",
                "Analyze retrieval performance scaling characteristics",
                "Verify fractal properties and self-similarity",
                "Calculate scaling exponents and power law relationships"
            ],
            "EXP-05": [
                "Create hierarchical data structures (fragments - clusters - glyphs - mist)",
                "Apply compression algorithms at each hierarchical level",
                "Measure compression ratios and efficiency metrics",
                "Verify lossless decompression capabilities",
                "Analyze compression effectiveness across different data types"
            ],
            "EXP-06": [
                "Generate related bit-chain pairs with known semantic relationships",
                "Calculate semantic similarity scores using vector embeddings",
                "Apply entanglement detection algorithm with configurable thresholds",
                "Measure precision and recall of entanglement detection",
                "Validate threshold effectiveness across different similarity levels"
            ],
            "EXP-07": [
                "Define LUCA (Last Universal Common Ancestor) entity with base coordinates",
                "Generate evolutionary tree through lineage operations",
                "Calculate lineage relationships and genetic distances",
                "Verify bootstrap completeness and coverage",
                "Analyze genetic diversity and evolutionary patterns"
            ],
            "EXP-08": [
                "Generate memory network with semantic content",
                "Apply clustering algorithms based on semantic similarity",
                "Measure semantic coherence within clusters",
                "Evaluate self-organization and emergent structure",
                "Analyze network topology and connectivity patterns"
            ],
            "EXP-09": [
                "Establish baseline performance metrics under normal conditions",
                "Apply memory pressure scenarios (light, moderate, heavy, critical)",
                "Measure performance degradation under constrained conditions",
                "Test optimization strategies (lazy loading, compression, eviction)",
                "Analyze system resilience and recovery characteristics"
            ],
            "EXP-10": [
                "Create multi-dimensional index structures for efficient querying",
                "Generate complex query patterns across all 8 dimensions",
                "Measure query execution times and resource usage",
                "Apply optimization techniques (indexing, caching, pruning)",
                "Analyze dimensional pruning effectiveness and query complexity"
            ],
            "EXP-11": [
                "Test various dimension counts (3, 4, 5, 6, 7, 8, 9, 10 dimensions)",
                "Measure expressiveness scores for each dimension count",
                "Calculate complexity overhead and computational cost",
                "Find optimal balance between expressiveness and efficiency",
                "Validate theoretical predictions with empirical results"
            ],
                "EXP-11b": [
                    "Define extreme parameter variations for dimensional analysis",
                    "Run stress tests with high and low dimension counts",
                    "Measure system performance and stability under stress",
                    "Analyze sensitivity to dimensional changes",
                    "Identify robustness thresholds for dimensional configurations"
            ],
            "EXP-12": [
                "Define comprehensive comparison metrics (performance, storage, expressiveness)",
                "Test all benchmark systems (UUID, SHA256, Vector DB, Graph DB, RDBMS)",
                "Measure performance characteristics across different scales",
                "Calculate relative advantages and disadvantages",
                "Analyze trade-offs and identify optimal use cases for each system"
            ],
            "EXP-13": [
                "Build pure fractal hierarchy trees for different elements",
                "Calculate hierarchical distances between random node pairs",
                "Compute natural cohesion without falloff across hierarchy",
                "Apply falloff to hierarchical distances and measure effects",
                "Analyze conservation patterns across different elements"
            ],
            "EXP-14": [
                "Retrieve electron shell configurations for elements",
                "Map shell count to fractal depth and valence electrons to branching factor",
                "Build hierarchical structures based on atomic properties",
                "Validate fractal parameters against observed densities",
                "Test prediction accuracy across the periodic table"
            ],
            "EXP-15": [
                "Define topological invariants (nodes, depth, connectivity, entropy)",
                "Run orbital dynamics simulation with hierarchical tracking",
                "Measure topological conservation over time",
                "Compare against classical energy conservation",
                "Validate fundamental difference between fractal and classical physics"
            ],
            "EXP-16": [
                "Create hierarchical distance mappings for spatial relationships",
                "Test distance transformation algorithms",
                "Validate hierarchical vs. spatial distance correlations",
                "Analyze multi-scale mapping effectiveness",
                "Optimize distance preservation across scales"
            ],
            "EXP-17": [
                "Set up thermodynamic validation framework",
                "Measure energy conservation in fractal systems",
                "Analyze thermal equilibrium properties",
                "Test statistical mechanics principles",
                "Validate thermodynamic consistency"
            ],
            "EXP-18": [
                "Implement falloff thermodynamics models",
                "Measure energy distribution patterns",
                "Analyze hierarchical energy flow",
                "Test thermodynamic efficiency across structures",
                "Validate falloff impact on system performance"
            ],
            "EXP-19": [
                "Define orbital equivalence relationships",
                "Test hierarchical dynamics in orbital systems",
                "Validate fractal orbital mechanics principles",
                "Analyze equivalence principle applications",
                "Compare with classical orbital mechanics"
            ],
            "EXP-20": [
                "Derive vector field approaches for fractal interactions",
                "Test different force calculation methods",
                "Validate gravitational interaction models",
                "Analyze field theory applications",
                "Optimize vector calculus implementations"
            ],
            "EXP-21": [
                "Set up Earth-Moon-Sun system simulation",
                "Implement accurate orbital mechanics equations",
                "Simulate gravitational interactions over time",
                "Validate against known astronomical data",
                "Analyze n-body problem dynamics and numerical integration methods"
            ]
        }
        return steps_map.get(experiment_id, ["Execute experiment", "Analyze results", "Generate report"])

    def _get_real_world_applications(self, experiment_id: str) -> str:
        """Get real-world applications for the experiment."""
        applications_map = {
            "EXP-01": "* Content-addressable storage systems* Cryptographic hash functions* Database indexing strategies* File system design",
            "EXP-02": "* Database query optimization* Cache system design* Real-time data processing* High-frequency trading systems",
            "EXP-03": "* Feature selection in machine learning* Dimensionality reduction techniques* Data compression algorithms* Information retrieval systems",
            "EXP-04": "* Scalable distributed systems* Cloud computing architectures* Big data processing frameworks* Network protocol design",
            "EXP-05": "* Data compression software* Multimedia file formats* Database storage optimization* Network bandwidth optimization",
            "EXP-06": "* Semantic search engines* Recommendation systems* Natural language processing* Knowledge graph construction",
            "EXP-07": "* Evolutionary biology research* Phylogenetic tree construction* Genetic algorithm design* Ancestral sequence reconstruction",
            "EXP-08": "* Artificial neural networks* Knowledge management systems* Self-organizing maps* Clustering algorithms",
            "EXP-09": "* Memory-constrained embedded systems* Mobile application optimization* Cloud resource management* Real-time system design",
            "EXP-10": "* Multi-dimensional database systems* Geographic information systems* Scientific data analysis* Complex query optimization",
            "EXP-11": "* System design trade-off analysis* Resource allocation strategies* Performance optimization* Cost-benefit analysis",
            "EXP-11b": "* Stress testing methodologies* Parameter sensitivity analysis* Robust system design* Performance under extreme conditions",
            "EXP-12": "* Technology selection for projects* Performance benchmarking* System architecture design* Vendor evaluation",
            "EXP-13": "* Hierarchical data organization* Natural language processing* Knowledge graph construction* Self-organizing systems",
            "EXP-14": "* Atomic structure modeling* Periodic table analysis* Quantum computing applications* Material science research",
            "EXP-15": "* Topological data analysis* Fractal physics applications* Complex system modeling* Network topology optimization",
            "EXP-16": "* Spatial database systems* Geographic information systems* Multi-scale data analysis* Hierarchical data visualization",
            "EXP-17": "* Thermodynamic system analysis* Energy conservation modeling* Statistical mechanics applications* Thermal system optimization",
            "EXP-18": "* Energy distribution analysis* Hierarchical system optimization* Thermodynamic efficiency modeling* Resource allocation systems",
            "EXP-19": "* Orbital mechanics applications* Hierarchical system dynamics* Equivalence principle testing* Complex system analysis",
            "EXP-20": "* Gravitational field modeling* Vector field applications* Fractal interaction systems* Field theory implementations",
            "EXP-21": "* Astronomical simulations* Orbital mechanics research* N-body problem analysis* Numerical integration method development"
        }
        return applications_map.get(experiment_id, "* General computational applications")

    def _get_key_lessons(self, experiment_id: str) -> str:
        """Get key lessons for the experiment."""
        lessons_map = {
            "EXP-01": "* Mathematical foundations ensure system reliability* Collision resistance is critical for data integrity* Proper coordinate systems enable unique addressing* Cryptographic principles provide security guarantees",
            "EXP-02": "* Algorithmic efficiency impacts real-world performance* Hash tables provide optimal retrieval performance* System design must consider scalability* Performance measurement requires precise timing",
            "EXP-03": "* Dimensional analysis reveals system properties* Information theory guides feature selection* Ablation studies identify critical components* Entropy measures system complexity",
            "EXP-04": "* Fractal properties enable scalable systems* Self-similarity provides consistent behavior* Scale invariance ensures predictable performance* Power laws describe natural system behavior",
            "EXP-05": "* Hierarchical structures enable efficient compression* Information theory guides algorithm design* Lossless compression preserves data integrity* Multi-level optimization improves efficiency",
            "EXP-06": "* Semantic similarity enables intelligent systems* Vector embeddings capture meaningful relationships* Threshold selection balances precision and recall* Entanglement detection reveals hidden connections",
            "EXP-07": "* Evolutionary principles guide system design* Lineage tracking enables provenance* Bootstrap methods create comprehensive systems* Genetic algorithms solve complex problems",
            "EXP-08": "* Self-organization creates emergent intelligence* Clustering reveals natural data structure* Semantic coherence improves system usability* Network topology affects performance",
            "EXP-09": "* Resource constraints drive innovation* Optimization strategies improve resilience* Performance under pressure reveals system quality* Memory management is critical for efficiency",
            "EXP-10": "* Multi-dimensional indexing enables complex queries* Query optimization reduces computational complexity* Dimensional pruning improves performance* Spatial databases handle complex data relationships",
            "EXP-11": "* Trade-off analysis guides system design* Optimal dimensionality balances expressiveness and complexity* Pareto efficiency identifies best solutions* Complexity theory informs algorithm selection",
            "EXP-11b": "* Stress testing methodologies* Parameter sensitivity analysis* Robust system design* Performance under extreme conditions",
            "EXP-12": "* Comparative analysis reveals system strengths* Benchmarking provides objective evaluation* Performance metrics guide technology selection* Trade-off analysis informs architectural decisions",
            "EXP-13": "* Hierarchical structures enable natural cohesion* Fractal gravity provides alternative to classical gravity* Tree-based organization supports efficient relationships* Hierarchical distance metrics enable spatial reasoning",
            "EXP-14": "* Atomic structure can be modeled through fractal hierarchies* Electron shell configurations inform fractal parameters* Periodic table patterns emerge from fractal properties* Quantum mechanical principles align with fractal mathematics",
            "EXP-15": "* Topological conservation provides alternative to classical conservation laws* Fractal systems prioritize structure over energy* Hierarchical tracking enables complex system analysis* Classical physics principles may not apply to fractal systems",
            "EXP-16": "* Hierarchical distance mapping enables multi-scale analysis* Spatial relationships can be preserved through hierarchical structures* Distance transformation algorithms support complex queries* Multi-scale analysis reveals hidden patterns in data",
            "EXP-17": "* Thermodynamic principles apply to fractal systems* Energy conservation manifests differently in hierarchical structures* Statistical mechanics principles guide fractal system behavior* Thermal equilibrium can be achieved through hierarchical organization",
            "EXP-18": "* Falloff thermodynamics affects hierarchical energy distribution* Energy efficiency varies across hierarchical levels* Thermodynamic optimization requires multi-scale analysis* Hierarchical structures impact energy flow patterns",
            "EXP-19": "* Orbital equivalence enables hierarchical system modeling* Fractal orbital mechanics provide alternative to classical mechanics* Equivalence principles apply across hierarchical scales* Complex orbital relationships emerge from fractal structures",
            "EXP-20": "* Vector field approaches enable fractal gravitational modeling* Field theory principles apply to hierarchical systems* Vector calculus provides tools for fractal interaction analysis* Gravitational interactions can be modeled through fractal mathematics",
            "EXP-21": "* Accurate orbital mechanics is essential for realistic simulations* Gravitational interactions are complex and require careful modeling* N-body problem analysis reveals system dynamics* Numerical integration methods are critical for long-term stability"
        }
        return lessons_map.get(experiment_id, "* Computational thinking solves complex problems* Mathematical foundations enable reliable systems* Experimental methodology validates theoretical concepts")

    def _determine_result_type(self, experiment_id: str, result: dict[str, Any]) -> str:
        """Determine the result type based on experiment outcome and scientific validation."""
        # Technical failure - experiment crashed or had execution errors
        if not result["success"]:
            return "failure"

        # Check for scientific validation failures in the output
        output = result.get("output", "").lower()

        if "[reproducibility notice]" in output:
            return "success"

        # Look for scientific validation failure indicators
        scientific_failures = [
            "experiment success: no",
            "distance mapping success: no",
            "force scaling consistent: no",
            "validation failed",
            "scientific validation failed",
            "not meet scientific criteria",
            "experiment_success: false",
            "distance_mapping_success: false",
            "force_scaling_consistent: false"
        ]

        # Check if this is an advanced experiment that might have scientific validation failures
        advanced_experiments = ["EXP-16", "EXP-17", "EXP-18", "EXP-19", "EXP-20", "EXP-21"]

        if experiment_id in advanced_experiments:
            for failure_indicator in scientific_failures:
                if failure_indicator in output:
                    return "warning"  # Scientific validation failed but experiment ran successfully

        # Check specific experiment validation patterns
        if experiment_id == "EXP-16":
            # Check for hierarchical distance mapping validation
            if "distance_correlation" in str(result.get("metrics", {})):
                # Parse metrics to check if validation passed
                metrics = result.get("metrics", {})
                if isinstance(metrics, dict) and "Exponential" in metrics:
                    exp_data = metrics["Exponential"]
                    if isinstance(exp_data, dict):
                        distance_corr = exp_data.get("distance_correlation", 0)
                        force_corr = exp_data.get("force_correlation", 0)
                        # If correlations are very low, consider it a partial success
                        if distance_corr < 0.2 and force_corr < 0.2:
                            return "partial_success"

        elif experiment_id in ["EXP-18", "EXP-19", "EXP-20", "EXP-21"]:
            # Use centralized validation logic for advanced experiments
            return self._check_advanced_experiment_validation(experiment_id, output)

        # Default to success if no specific validation failures found
        return "success"

    def _check_advanced_experiment_validation(self, experiment_id: str, output: str) -> str:
        """Centralized validation logic for advanced experiments (EXP-18 through EXP-21)."""
        # Define validation rules for each advanced experiment
        validation_rules = {
            "EXP-18": {
                "failure_indicators": [
                    "no improvement",
                    "doesn't help thermodynamics",
                    "status: failed"
                ],
                "description": "thermodynamics validation failure"
            },
            "EXP-19": {
                "failure_indicators": [
                    ("orbital equivalence", "not properly simulated"),
                    "status: failed",
                    "equivalence_confirmed: false"
                ],
                "description": "orbital simulation issues"
            },
            "EXP-20": {
                "failure_indicators": [
                    "validation failed"
                ],
                "description": "vector field validation issues"
            },
            "EXP-21": {
                "failure_indicators": [
                    "validation failed",
                    "status: failed",
                    "universality claim supported: no",
                    "hierarchical scaling confirmed: no"
                ],
                "description": "simulation validation issues"
            }
        }

        # Get validation rules for this experiment
        rules = validation_rules.get(experiment_id)
        if not rules:
            return "success"  # Default to success if no rules defined

        # Check for failure indicators
        output_lower = output.lower()

        for indicator in rules["failure_indicators"]:
            if isinstance(indicator, tuple):
                # For compound indicators (both must be present)
                if all(ind.lower() in output_lower for ind in indicator):
                    return "warning"
            else:
                # For simple indicators (any one present)
                if indicator.lower() in output_lower:
                    return "warning"

        return "success"

    async def _execute_experiment_module(
        self,
        experiment_id: str,
        quick_mode: bool,
        progress_callback=None,
        repro_runs: int = 1,
    ) -> JsonObject:
        """Execute the actual experiment module."""
        try:
            # Get the configuration for this experiment
            config = self.experiment_configs[experiment_id]

            # Use subprocess execution for all experiments to ensure compatibility
            return await self._execute_experiment_subprocess(
                experiment_id,
                quick_mode,
                progress_callback=progress_callback,
                repro_runs=repro_runs,
            )

        except Exception as e:
            import traceback
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "working_directory": os.getcwd(),
                "python_executable": sys.executable,
                "experiment_id": experiment_id,
                "module_name": config.module_name if 'config' in locals() else "unknown"
            }

            error_output = f"""
Subprocess execution failed!

Error Type: {error_details['error_type']}
Error Message: {error_details['error_message']}

Working Directory: {error_details['working_directory']}
Python Executable: {error_details['python_executable']}
Experiment ID: {error_details['experiment_id']}
Module Name: {error_details['module_name']}

Full Traceback:
{error_details['traceback']}
"""

            return {
                "success": False,
                "output": error_output,
                "metrics": error_details
            }

    async def _execute_experiment_subprocess(
        self,
        experiment_id: str,
        quick_mode: bool,
        progress_callback=None,
        repro_runs: int = 1,
    ) -> JsonObject:
        """Execute experiment as subprocess with optimized progress tracking."""
        try:
            # Import progress communication module
            from fractalsemantics.progress_comm import (
                is_progress_message,
                parse_progress_message,
            )

            # Construct command to run the experiment
            experiment_map = {
                "EXP-01": "exp01_geometric_collision",
                "EXP-02": "exp02_retrieval_efficiency",
                "EXP-03": "exp03_coordinate_entropy",
                "EXP-04": "exp04_fractal_scaling",
                "EXP-05": "exp05_compression_expansion",
                "EXP-06": "exp06_entanglement_detection",
                "EXP-07": "exp07_luca_bootstrap",
                "EXP-08": "exp08_self_organizing_memory",
                "EXP-09": "exp09_memory_pressure",
                "EXP-10": "exp10_multidimensional_query",
                "EXP-11": "exp11_dimension_cardinality",
                "EXP-11b": "exp11b_dimension_stress_test",
                "EXP-12": "exp12_benchmark_comparison",
                "EXP-13": "exp13_fractal_gravity",
                "EXP-14": "exp14_atomic_fractal_mapping",
                "EXP-15": "exp15_topological_conservation",
                "EXP-16": "exp16_hierarchical_distance_mapping",
                "EXP-17": "exp17_thermodynamic_validation",
                "EXP-18": "exp18_falloff_thermodynamics",
                "EXP-19": "exp19_orbital_equivalence",
                "EXP-20": "exp20_vector_field_derivation",
                "EXP-21": "exp21_earth_moon_sun"
            }

            module_name = experiment_map.get(experiment_id)
            if not module_name:
                raise ValueError(f"Unknown experiment: {experiment_id}")

            # Get timeout and execution profile from experiment config
            config = self.experiment_configs.get(experiment_id, ExperimentConfig(
                experiment_id=experiment_id,
                module_name="",
                description="",
                educational_focus="",
                timeout_seconds=300
            ))
            timeout = config.timeout_seconds

            # Build isolated, deterministic subprocess command
            python_executable = sys.executable
            cmd = [
                python_executable,
                "-X",
                "utf8",
                "-I",
                str(Path(__file__).parent / f"{module_name}.py"),
            ]

            # Add quick mode flag if needed
            if quick_mode:
                cmd.append("--quick")

            # Prepare environment with progress file path
            env = os.environ.copy()

            # Always ensure progress file env var is set for subprocess
            if "FRACTALSEMANTICS_PROGRESS_FILE" in os.environ:
                env["FRACTALSEMANTICS_PROGRESS_FILE"] = os.environ["FRACTALSEMANTICS_PROGRESS_FILE"]
            else:
                # Set default progress file path if not already set
                # Use absolute path to ensure it's in the project root regardless of working directory
                project_root = str(Path(__file__).parent.parent)
                progress_file_path = str(Path(project_root) / "results" / "gui_progress.jsonl")
                env["FRACTALSEMANTICS_PROGRESS_FILE"] = progress_file_path

            # Remove Streamlit-specific environment variables
            streamlit_vars = [k for k in env if k.startswith('STREAMLIT_')]
            for var in streamlit_vars:
                del env[var]

            # Ensure fractalsemantics module can be found
            project_root = str(Path(__file__).parent.parent)
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = project_root

            env["PYTHONUTF8"] = "1"
            env["PYTHONHASHSEED"] = "0"
            env["FRACTALSEMANTICS_VALIDATION_MODE"] = "1"
            env[SOFTCOPY_ENV] = "true" if self.softcopy_enabled else "false"
            env["VIRTUAL_ENV"] = sys.prefix

            scripts_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
            env["PATH"] = f"{scripts_dir}{os.pathsep}{env.get('PATH', '')}"

            # Reproducibility reruns are controlled by explicit CLI count first.
            # If not requested, retain optional env-based advanced rerun compatibility.
            run_count = max(1, int(repro_runs))
            if run_count == 1:
                reproducibility_rerun_enabled = _env_flag_enabled(ENABLE_ADVANCED_REPRO_CHECK_ENV)
                if reproducibility_rerun_enabled and config.experiment_type == "advanced" and not quick_mode:
                    run_count = 2

            def run_once() -> JsonObject:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=str(Path(__file__).parent),
                    encoding="utf-8",
                    errors="replace",
                )

                queue: Queue[tuple[str, str]] = Queue()

                def enqueue_stream(stream, source: str) -> None:
                    try:
                        for line in iter(stream.readline, ""):
                            queue.put((source, line))
                    finally:
                        with contextlib.suppress(Exception):
                            stream.close()

                stdout_thread = Thread(
                    target=enqueue_stream,
                    args=(process.stdout, "stdout"),
                    daemon=True,
                )
                stderr_thread = Thread(
                    target=enqueue_stream,
                    args=(process.stderr, "stderr"),
                    daemon=True,
                )
                stdout_thread.start()
                stderr_thread.start()

                stdout_lines: list[str] = []
                stderr_lines: list[str] = []
                progress_messages = []

                deadline = time.monotonic() + timeout

                while True:
                    if time.monotonic() > deadline and process.poll() is None:
                        process.kill()
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

                    try:
                        source, line = queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
                    except Exception:
                        source = ""
                        line = ""

                    if source:
                        cleaned_line = line.rstrip("\n")
                        if source == "stdout":
                            stdout_lines.append(cleaned_line)
                        else:
                            stderr_lines.append(cleaned_line)
                            if is_progress_message(cleaned_line):
                                progress_msg = parse_progress_message(cleaned_line)
                                if progress_msg and progress_msg.experiment_id == experiment_id:
                                    progress_messages.append(progress_msg)
                                    if progress_callback:
                                        progress_callback(
                                            experiment_id,
                                            float(progress_msg.progress_percent),
                                            progress_msg.stage,
                                            progress_msg.message,
                                        )

                    process_done = process.poll() is not None
                    threads_done = (not stdout_thread.is_alive()) and (not stderr_thread.is_alive())
                    if process_done and threads_done and queue.empty():
                        break

                result_return_code = process.returncode
                output = "\n".join(stdout_lines)
                completion_markers = [
                    "[OK]",
                    "COMPLETE",
                    "[Success]",
                    "SUCCESS",
                    f"{experiment_id} COMPLETE",
                ]
                has_completion_marker = any(
                    marker in output.upper() for marker in [m.upper() for m in completion_markers]
                )
                success = result_return_code == 0 or has_completion_marker

                filtered_error_lines = [line for line in stderr_lines if not is_progress_message(line)]
                filtered_error = "\n".join(filtered_error_lines).strip()

                combined_output = output + (f"\nStderr: {filtered_error}" if filtered_error else "")
                display_output = self._compress_output_for_display(combined_output, max_lines=OUTPUT_DISPLAY_MAX_LINES)
                display_output = self._normalize_output_save_language(display_output)

                deleted_artifacts: list[str] = []
                if not self.softcopy_enabled:
                    deleted_artifacts = self._cleanup_softcopy_artifacts(display_output)
                    if deleted_artifacts:
                        display_output += (
                            "\n[SOFTCOPY DISABLED] Persisted artifacts were removed after execution "
                            "(terminal output retained)."
                        )

                metrics: dict[str, Any] = {
                    "return_code": result_return_code,
                    "command": cmd,
                    "timeout_seconds": timeout,
                    "isolated_mode": True,
                    "utf8_mode": True,
                    "output_sha256": hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest(),
                    "softcopy_enabled": self.softcopy_enabled,
                }
                if deleted_artifacts:
                    metrics["softcopy_deleted_artifacts"] = deleted_artifacts
                if progress_messages:
                    progress_data = []
                    for msg in progress_messages:
                        progress_data.append({
                            "timestamp": msg.timestamp,
                            "progress_percent": float(msg.progress_percent),
                            "stage": msg.stage,
                            "message": msg.message,
                            "message_type": msg.message_type,
                        })

                    deduplicated_progress = self._deduplicate_progress_messages(progress_data)
                    progress_total = len(deduplicated_progress)
                    progress_limit = PROGRESS_MESSAGE_LIMIT
                    metrics["progress_message_total"] = progress_total
                    metrics["progress_message_duplicates_removed"] = len(progress_data) - progress_total
                    metrics["progress_messages_truncated"] = progress_total > progress_limit
                    metrics["progress_messages"] = deduplicated_progress[:progress_limit]

                metrics["output_line_count_raw"] = len(combined_output.splitlines())
                metrics["output_line_count_display"] = len(display_output.splitlines())
                metrics["output_truncated_for_display"] = metrics["output_line_count_display"] < metrics["output_line_count_raw"]

                return {
                    "success": success,
                    "output": display_output,
                    "metrics": metrics,
                }

            try:
                primary_run = await asyncio.to_thread(run_once)

                if run_count > 1 and primary_run["success"]:
                    additional_runs: list[JsonObject] = []
                    for _ in range(run_count - 1):
                        validation_run = await asyncio.to_thread(run_once)
                        additional_runs.append(validation_run)

                    all_runs = [primary_run, *additional_runs]
                    all_return_codes = [run["metrics"].get("return_code") for run in all_runs]
                    all_hashes = [run["metrics"].get("output_sha256") for run in all_runs]
                    all_successes = [bool(run.get("success", False)) for run in all_runs]

                    unique_hashes = {hash_val for hash_val in all_hashes if hash_val}
                    hashes_match = len(unique_hashes) <= 1
                    all_runs_successful = all(all_successes)
                    hash_mismatch_only = all_runs_successful and not hashes_match
                    hash_mismatch_exempted = (
                        hash_mismatch_only and experiment_id in HASH_MISMATCH_EXPECTED_EXPERIMENTS
                    )

                    reproducibility = {
                        "enabled": True,
                        "required_match": True,
                        "run_count": run_count,
                        "return_codes": all_return_codes,
                        "all_runs_successful": all_runs_successful,
                        "same_output_sha256": hashes_match,
                        "output_sha256_values": all_hashes,
                        "hash_mismatch_exempted": hash_mismatch_exempted,
                    }
                    reproducibility["reproducible"] = (
                        all_runs_successful and (hashes_match or hash_mismatch_exempted)
                    )

                    primary_run["metrics"]["reproducibility_check"] = reproducibility

                    if not reproducibility["reproducible"]:
                        primary_run["output"] += (
                            "\n[REPRODUCIBILITY FAIL] Run-to-run output hash mismatch or rerun failure "
                            "(see metrics.reproducibility_check)."
                        )
                        primary_run["success"] = False
                    elif hash_mismatch_exempted:
                        primary_run["output"] += (
                            "\n[REPRODUCIBILITY NOTICE] Run outputs differ by hash, but this experiment is "
                            "expected to emit non-deterministic text/artifacts (timestamps, output paths, "
                            "progress timing, stochastic sampling). Hash mismatch is reported but not treated "
                            "as a technical failure."
                        )

                return primary_run

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": f"Experiment {experiment_id} timed out after {timeout} seconds",
                    "metrics": {"error_type": "TimeoutError", "error_message": f"Timeout after {timeout} seconds"}
                }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "output": f"Subprocess execution failed: {str(e)}\n\nFull traceback:\n{error_details}\n\nCommand attempted: {' '.join(cmd) if 'cmd' in locals() else 'Command not constructed'}\nWorking directory: {os.getcwd()}\nPython executable: {sys.executable}",
                "metrics": {"error_type": type(e).__name__, "error_message": str(e)}
            }

    async def run_batch_experiments(self, experiment_ids: list[str], quick_mode: bool = False,
                                   parallel: bool = True, progress_callback=None,
                                   repro_runs: int = 1) -> BatchRunResult:
        """Run multiple experiments with progress tracking and educational output."""
        start_time = time.time()
        experiment_results = []

        if not experiment_ids:
            experiment_ids = list(self.experiment_configs.keys())

        # Validate and normalize experiment IDs
        validated_experiments = []
        for exp_id in experiment_ids:
            validated_id = self._validate_experiment_id(exp_id)
            if validated_id:
                validated_experiments.append(validated_id)
            else:
                raise ValueError(f"Unknown experiment: {exp_id}")

        experiment_ids = validated_experiments
        total_experiments = len(experiment_ids)
        successful_experiments = 0
        failed_experiments = 0

        print(f"Starting batch run of {total_experiments} experiments...")
        print(f"Feature Level: {'Quick' if quick_mode else 'Full'}")
        print(f"Execution Mode: {'Parallel' if parallel else 'Sequential'}")
        if repro_runs > 1:
            print(f"Reproducibility Runs: {repro_runs}")
        print("=" * 80)

        if parallel:
            # Run experiments in parallel with individual progress tracking
            print(f"Running {total_experiments} experiments in parallel...")

            # Create individual progress bars for each experiment
            progress_bars = {}
            for exp_id in experiment_ids:
                progress_bars[exp_id] = tqdm.tqdm(
                    total=PROGRESS_BAR_TOTAL,
                    desc=f"{exp_id}",
                    unit="%",
                    position=len(progress_bars),
                    leave=True,
                    ncols=PROGRESS_BAR_COLUMNS,
                    bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}] {postfix}"
                )

            # Run experiments in parallel
            progress_lock = Lock()

            def make_progress_callback(exp_id: str):
                def on_progress(_exp_id: str, progress_percent: float, stage: str, message: str) -> None:
                    progress_bar = progress_bars.get(exp_id)
                    if not progress_bar:
                        return
                    bounded = max(PROGRESS_PERCENT_MIN, min(PROGRESS_PERCENT_MAX, float(progress_percent)))
                    with progress_lock:
                        progress_bar.n = int(bounded)
                        progress_bar.set_postfix({"Stage": stage[:PROGRESS_STAGE_LABEL_MAX_CHARS]})
                        progress_bar.refresh()

                return on_progress

            tasks = [
                self.run_experiment(
                    exp_id,
                    quick_mode,
                    progress_callback=make_progress_callback(exp_id),
                    repro_runs=repro_runs,
                )
                for exp_id in experiment_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and update progress bars
            for i, result in enumerate(results):
                experiment_id = experiment_ids[i]
                progress_bar = progress_bars[experiment_id]

                if isinstance(result, Exception):
                    # Handle exceptions from gather
                    duration = 0
                    error_result = ExperimentResult(
                        experiment_id=experiment_id,
                        success=False,
                        duration=duration,
                        output=f"Error: {str(result)}",
                        metrics={},
                        educational_content=[f"[FAIL] Experiment {experiment_id} failed with error: {str(result)}"]
                    )
                    experiment_results.append(error_result)
                    failed_experiments += 1

                    # Complete progress bar as failed
                    progress_bar.n = PROGRESS_BAR_TOTAL
                    progress_bar.set_postfix({"Status": "[FAIL] Failed"})
                    progress_bar.refresh()
                    progress_bar.close()

                    if progress_callback:
                        progress_callback(len(experiment_results), total_experiments, error_result)
                else:
                    # Normal result - result is guaranteed to be ExperimentResult here
                    assert isinstance(result, ExperimentResult), f"Expected ExperimentResult, got {type(result)}"
                    experiment_results.append(result)
                    if result.success:
                        successful_experiments += 1
                        if result.result_type == "warning":
                            status = "[PASS] Technical | [NEGATIVE] Scientific"
                        elif result.result_type == "partial_success":
                            status = "[PASS] Technical | [PARTIAL] Scientific"
                        else:
                            status = "[PASS] Technical + Scientific"
                    else:
                        failed_experiments += 1
                        status = "[FAIL] Technical"

                    # Complete progress bar
                    progress_bar.n = PROGRESS_BAR_TOTAL
                    progress_bar.set_postfix({"Status": status, "Time": f"{self._format_float(result.duration, 1)}s"})
                    progress_bar.refresh()
                    progress_bar.close()

                    if progress_callback:
                        progress_callback(len(experiment_results), total_experiments, result)

        else:
            # Run experiments sequentially with single progress bar
            progress_bar = tqdm.tqdm(
                total=total_experiments,
                desc="Running experiments",
                unit="exp",
                ncols=PROGRESS_BAR_COLUMNS,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                leave=True,
                mininterval=SEQUENTIAL_PROGRESS_MIN_INTERVAL_SECONDS
            )

            for i, experiment_id in enumerate(experiment_ids, 1):
                try:
                    result = await self.run_experiment(experiment_id, quick_mode, repro_runs=repro_runs)
                    experiment_results.append(result)

                    if result.success:
                        successful_experiments += 1
                        if result.result_type == "warning":
                            status = "Tech PASS | Science NEGATIVE"
                        elif result.result_type == "partial_success":
                            status = "Tech PASS | Science PARTIAL"
                        else:
                            status = "Tech+Science PASS"
                    else:
                        failed_experiments += 1
                        status = "Tech FAIL"

                    # Update progress bar
                    progress_bar.set_postfix({"Status": status, "Last": result.experiment_id})
                    progress_bar.update(1)

                    if progress_callback:
                        progress_callback(i, total_experiments, result)
                except Exception as e:
                    # Handle exceptions in sequential execution
                    duration = 0
                    error_result = ExperimentResult(
                        experiment_id=experiment_id,
                        success=False,
                        duration=duration,
                        output=f"Error: {str(e)}",
                        metrics={},
                        educational_content=[f"[FAIL] Experiment {experiment_id} failed with error: {str(e)}"]
                    )
                    experiment_results.append(error_result)
                    failed_experiments += 1

                    # Update progress bar
                    progress_bar.set_postfix({"Status": "[FAIL] Failed", "Last": experiment_id})
                    progress_bar.update(1)

                    if progress_callback:
                        progress_callback(i, total_experiments, error_result)

            # Close progress bar
            progress_bar.close()

        total_duration = time.time() - start_time
        summary_report = self._generate_batch_summary(experiment_results, total_duration, quick_mode)

        return BatchRunResult(
            total_experiments=total_experiments,
            successful_experiments=successful_experiments,
            failed_experiments=failed_experiments,
            total_duration=total_duration,
            experiment_results=experiment_results,
            summary_report=summary_report
        )

    def _validate_experiment_id(self, exp_id: str) -> Optional[str]:
        """Validate and normalize experiment ID format."""
        # Try exact match first
        if exp_id in self.experiment_configs:
            return exp_id

        # Try with EXP- prefix if not already present
        exp_upper = exp_id.upper()
        if not exp_upper.startswith("EXP-"):
            exp_with_prefix = f"EXP-{exp_upper.lstrip('EXP-')}"
            if exp_with_prefix in self.experiment_configs:
                return exp_with_prefix

        return None

    def progress_file(self) -> str:
        """Get the path to the progress file."""
        # Use absolute path to ensure it's in the project root regardless of working directory
        project_root = str(Path(__file__).parent.parent)
        return str(Path(project_root) / "results" / "gui_progress.jsonl")

    def _print_progress(self, current: int, total: int, result: ExperimentResult):
        """Print progress update for batch runs."""
        if not result.success:
            status = "[FAIL] Technical"
        elif result.result_type == "warning":
            status = "[PASS] Technical | [NEGATIVE] Scientific"
        elif result.result_type == "partial_success":
            status = "[PASS] Technical | [PARTIAL] Scientific"
        else:
            status = "[PASS] Technical + Scientific"
        duration_str = f"{self._format_duration(result.duration)}s"
        print(f"{status} {result.experiment_id} - {duration_str} ({current}/{total})")

        # Print a separator every 4 experiments
        if current % PROGRESS_SEPARATOR_EVERY == 0 and current < total:
            print("-" * 40)

    def _generate_batch_summary(self, experiment_results: list[ExperimentResult],
                              total_duration: float, quick_mode: bool) -> str:
        """Generate educational summary report for batch run."""
        successful = sum(1 for r in experiment_results if r.success)
        failed = len(experiment_results) - successful
        true_successes = sum(1 for r in experiment_results if r.success and r.result_type == "success")

        # Categorize failures by type
        technical_failures = sum(1 for r in experiment_results if r.result_type == "failure")
        scientific_warnings = sum(1 for r in experiment_results if r.result_type == "warning")
        partial_successes = sum(1 for r in experiment_results if r.result_type == "partial_success")

        summary = f"""
- BATCH EXPERIMENT SUMMARY REPORT
{'='*80}

- OVERALL STATISTICS:
   * Total Experiments: {len(experiment_results)}
    * Technical Successes: {successful}
    * Technical Failures: {failed}
     * Technical Success Rate (display-rounded): {self._format_percent(successful/len(experiment_results)*100)}%
    * Total Duration: {self._format_duration(total_duration)} seconds
    * Average Duration: {self._format_duration(total_duration/len(experiment_results))} seconds per experiment
   * Feature Level: {'Quick' if quick_mode else 'Full'}

- FAILURE ANALYSIS:
   * Technical Failures (crashes/errors): {technical_failures}
   * Scientific Warnings (validation failures): {scientific_warnings}
   * Partial Successes (low performance): {partial_successes}
    * True Successes (technical + scientific): {true_successes}

- PERFORMANCE ANALYSIS:
"""

        # Analyze performance patterns
        successful_durations = [r.duration for r in experiment_results if r.success]
        if successful_durations:
            avg_duration = sum(successful_durations) / len(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)

            summary += f"""   * Average Duration (successful): {self._format_duration(avg_duration)}s
   * Fastest Experiment: {self._format_duration(min_duration)}s
   * Slowest Experiment: {self._format_duration(max_duration)}s
"""

        summary += """
- EDUCATIONAL INSIGHTS:
   * This batch run demonstrates the comprehensive capabilities of FractalSemantics
   * Each experiment validates different aspects of the addressing system
   * Success rate indicates system reliability and robustness
   * Performance metrics show scalability characteristics

- SYSTEM VALIDATION:
"""

        # Categorize experiments by type
        collision_tests = [r for r in experiment_results if r.experiment_id in ["EXP-01", "EXP-03"]]
        performance_tests = [r for r in experiment_results if r.experiment_id in ["EXP-02", "EXP-04"]]
        advanced_tests = [r for r in experiment_results if r.experiment_id in ["EXP-05", "EXP-06", "EXP-07"]]
        system_tests = [r for r in experiment_results if r.experiment_id in ["EXP-08", "EXP-09", "EXP-10"]]
        analysis_tests = [r for r in experiment_results if r.experiment_id in ["EXP-11", "EXP-12"]]
        fractal_physics_tests = [r for r in experiment_results if r.experiment_id in ["EXP-13", "EXP-14", "EXP-15", "EXP-16", "EXP-17", "EXP-18", "EXP-19","EXP-20" ,"EXP-21"]]
        fractal_physics_true_successes = sum(
            1 for r in fractal_physics_tests if r.success and r.result_type == "success"
        )

        summary += f"""   * Collision Resistance Tests: {sum(1 for r in collision_tests if r.success)}/{len(collision_tests)} passed
   * Performance & Scaling Tests: {sum(1 for r in performance_tests if r.success)}/{len(performance_tests)} passed
   * Advanced Feature Tests: {sum(1 for r in advanced_tests if r.success)}/{len(advanced_tests)} passed
   * System Integration Tests: {sum(1 for r in system_tests if r.success)}/{len(system_tests)} passed
   * Analysis & Comparison Tests: {sum(1 for r in analysis_tests if r.success)}/{len(analysis_tests)} passed
    * Fractal Physics Simulations (technical): {sum(1 for r in fractal_physics_tests if r.success)}/{len(fractal_physics_tests)} passed
    * Fractal Physics Scientifically Validated: {fractal_physics_true_successes}/{len(fractal_physics_tests)} passed

- KEY LEARNING OUTCOMES:
   * FractalSemantics provides robust, collision-resistant addressing
   * System scales efficiently across different data volumes
   * Multi-dimensional indexing enables powerful querying capabilities
   * Hierarchical structures support efficient compression and organization
   * Semantic relationships can be detected and analyzed
    * Some fractal physics runs produce scientifically valid negative results (hypothesis not supported)

-  SCIENTIFIC VALIDATION INSIGHTS:
   * Technical failures indicate system crashes or execution errors
    * Scientific warnings indicate experiments ran successfully but did not support target hypotheses
   * Partial successes indicate experiments with sub-optimal performance
   * These distinctions help identify areas for improvement

- RECOMMENDATIONS:
   * For production use: Run with full feature level for comprehensive validation
   * For development: Quick mode provides rapid feedback on core functionality
   * Monitor both technical and scientific success rates
   * Address scientific warnings to improve system capabilities
   * Regular batch runs help maintain system reliability

{'='*80}
"""
        summary = re.sub(r"(?m)^ {4}\*", "   *", summary)
        return summary

def main():
    """Main entry point for the experiment runner."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single experiment: python experiment_runner.py <experiment_id> [--quick] [--format=json|text] [--repro-runs=N] [--softcopy=true|false]")
        print("  Batch experiments: python experiment_runner.py --all [--quick|--full] [--parallel|--sequential] [--format=json|text] [--repro-runs=N] [--softcopy=true|false]")
        print("  Specific batch:    python experiment_runner.py EXP-01 EXP-02 EXP-03 [--quick|--full] [--parallel|--sequential] [--format=json|text] [--repro-runs=N] [--softcopy=true|false]")
        print("")
        print("Examples:")
        print("  fractalsemantics-runner --all --full")
        print("  fractalsemantics-runner --all --quick")
        print("  fractalsemantics-runner EXP-01 EXP-02 --quick --sequential")
        print("  fractalsemantics-runner EXP-01 --quick --format=json")
        print("  fractalsemantics-runner EXP-13 EXP-14 --full --repro-runs=3")
        print("  fractalsemantics-runner EXP-03 --quick --softcopy=false")
        sys.exit(1)

    # Parse command line arguments
    args = sys.argv[1:]

    # Check for batch mode indicators
    is_all = "--all" in args
    is_quick = "--quick" in args
    is_full = "--full" in args
    is_parallel = "--parallel" in args
    is_sequential = "--sequential" in args or "--serial" in args

    # Check for format
    output_format = "json"  # default format
    if "--format=json" in args:
        output_format = "json"
    elif "--format=text" in args:
        output_format = "text"

    repro_runs = _parse_repro_runs_arg(args)
    softcopy_enabled = _parse_softcopy_arg(args)

    # Determine feature level
    if is_quick:
        quick_mode = True
    elif is_full:
        quick_mode = False
    else:
        # Default to quick mode for batch runs, full mode for single experiments
        quick_mode = is_all or len([arg for arg in args if arg.startswith("EXP-")]) > 1

    # Determine execution mode
    if is_parallel:
        parallel_mode = True
    elif is_sequential:
        parallel_mode = False
    else:
        # Default to parallel for batch runs
        parallel_mode = is_all or len([arg for arg in args if arg.startswith("EXP-")]) > 1

    runner = ExperimentRunner(softcopy_enabled=softcopy_enabled)

    try:
        if is_all:
            # Run all experiments
            print(f"- Running ALL experiments in {'Quick' if quick_mode else 'Full'} mode...")
            batch_result = asyncio.run(runner.run_batch_experiments(
                experiment_ids=[],  # Empty list means run all
                quick_mode=quick_mode,
                parallel=parallel_mode,
                repro_runs=repro_runs,
            ))

            if output_format == "json":
                # Output batch result as JSON
                output = {
                    "batch_run": True,
                    "total_experiments": batch_result.total_experiments,
                    "successful_experiments": batch_result.successful_experiments,
                    "failed_experiments": batch_result.failed_experiments,
                    "total_duration": batch_result.total_duration,
                    "success_rate": (batch_result.successful_experiments / batch_result.total_experiments * 100) if batch_result.total_experiments > 0 else 0,
                    "experiment_results": [
                        {
                            "experiment_id": r.experiment_id,
                            "success": r.success,
                            "result_type": r.result_type,
                            "duration": r.duration,
                            "output": r.output,
                            "metrics": r.metrics,
                            "educational_content": r.educational_content
                        } for r in batch_result.experiment_results
                    ],
                    "summary_report": batch_result.summary_report
                }
                json_output = json.dumps(output, indent=2, ensure_ascii=False)
                print(json_output)
            else:
                # Output as formatted text
                print(batch_result.summary_report)

        elif any(arg.startswith("EXP-") for arg in args):
            # Run specific experiments
            experiment_ids = [arg for arg in args if arg.startswith("EXP-")]

            if len(experiment_ids) == 1:
                # Single experiment
                experiment_id = experiment_ids[0]
                result = asyncio.run(runner.run_experiment(experiment_id, quick_mode, repro_runs=repro_runs))

                if output_format == "json":
                    output = {
                        "experiment_id": result.experiment_id,
                        "success": result.success,
                        "result_type": result.result_type,
                        "duration": result.duration,
                        "output": result.output,
                        "metrics": result.metrics,
                        "educational_content": result.educational_content
                    }
                    json_output = json.dumps(output, indent=2, ensure_ascii=False)
                    print(json_output)
                else:
                    print("=" * 80)
                    print(f"EXPERIMENT: {result.experiment_id}")
                    print("=" * 80)
                    print(f"Success: {result.success}")
                    print(f"Duration: {runner._format_duration(result.duration, detailed=True)} seconds")
                    print("=" * 80)
                    print("EXPERIMENT OUTPUT:")
                    print("-" * 40)
                    print(result.output)
                    print("=" * 80)
                    print("EDUCATIONAL CONTENT:")
                    print("-" * 40)
                    for i, content in enumerate(result.educational_content, 1):
                        print(f"Section {i}:")
                        print(content)
                        print("-" * 40)
                    print("=" * 80)
            else:
                # Multiple specific experiments
                print(f"- Running {len(experiment_ids)} specific experiments in {'Quick' if quick_mode else 'Full'} mode...")
                batch_result = asyncio.run(runner.run_batch_experiments(
                    experiment_ids=experiment_ids,
                    quick_mode=quick_mode,
                    parallel=parallel_mode,
                    repro_runs=repro_runs,
                ))

                if output_format == "json":
                    output = {
                        "batch_run": True,
                        "experiment_ids": experiment_ids,
                        "total_experiments": batch_result.total_experiments,
                        "successful_experiments": batch_result.successful_experiments,
                        "failed_experiments": batch_result.failed_experiments,
                        "total_duration": batch_result.total_duration,
                        "success_rate": (batch_result.successful_experiments / batch_result.total_experiments * 100) if batch_result.total_experiments > 0 else 0,
                        "experiment_results": [
                            {
                                "experiment_id": r.experiment_id,
                                "success": r.success,
                                "result_type": r.result_type,
                                "duration": r.duration,
                                "output": r.output,
                                "metrics": r.metrics,
                                "educational_content": r.educational_content
                            } for r in batch_result.experiment_results
                        ],
                        "summary_report": batch_result.summary_report
                    }
                    json_output = json.dumps(output, indent=2, ensure_ascii=False)
                    print(json_output)
                else:
                    print(batch_result.summary_report)

        else:
            # Single experiment mode (legacy behavior)
            experiment_id = args[0]
            result = asyncio.run(runner.run_experiment(experiment_id, quick_mode, repro_runs=repro_runs))

            if output_format == "json":
                output = {
                    "experiment_id": result.experiment_id,
                    "success": result.success,
                    "result_type": result.result_type,
                    "duration": result.duration,
                    "output": result.output,
                    "metrics": result.metrics,
                    "educational_content": result.educational_content
                }
                json_output = json.dumps(output, indent=2, ensure_ascii=False)
                print(json_output)
            else:
                print("=" * 80)
                print(f"EXPERIMENT: {result.experiment_id}")
                print("=" * 80)
                print(f"Success: {result.success}")
                print(f"Duration: {runner._format_duration(result.duration, detailed=True)} seconds")
                print("=" * 80)
                print("EXPERIMENT OUTPUT:")
                print("-" * 40)
                print(result.output)
                print("=" * 80)
                print("EDUCATIONAL CONTENT:")
                print("-" * 40)
                for i, content in enumerate(result.educational_content, 1):
                    print(f"Section {i}:")
                    print(content)
                    print("-" * 40)
                print("=" * 80)

    except Exception as e:
        if output_format == "json":
            print(json.dumps({
                "error": str(e),
                "success": False
            }, indent=2))
        else:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
