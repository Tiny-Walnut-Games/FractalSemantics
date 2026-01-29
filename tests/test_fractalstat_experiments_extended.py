"""
Extended tests for FractalStat Experiments to achieve 95%+ coverage
Tests EXP-02, EXP-03, run_all_experiments(), and edge cases
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from fractalstat.dynamic_enum import Alignment, Polarity


class TestEXP02RetrievalEfficiency:
    """Tests for EXP-02 retrieval efficiency experiment."""

    def test_exp02_initialization(self):
        """EXP02_RetrievalEfficiency should initialize with query count."""
        from fractalstat.fractalstat_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=500)
        assert exp.query_count == 500
        assert len(exp.scales) > 0
        assert exp.results == []

    def test_exp02_run_small_scale(self):
        """EXP-02 run should execute with small scale for testing."""
        from fractalstat.fractalstat_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=100)
        # Override scales for faster testing
        exp.scales = [100]

        results, success = exp.run()

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(success, bool)
        assert results[0].scale == 100
        assert results[0].queries == 100

    def test_exp02_latency_calculations(self):
        """EXP-02 should calculate latency percentiles correctly."""
        from fractalstat.fractalstat_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=50)
        exp.scales = [50]

        results, _ = exp.run()

        result = results[0]
        assert result.mean_latency_ms >= 0
        assert result.median_latency_ms >= 0
        assert result.p95_latency_ms >= result.median_latency_ms
        assert result.p99_latency_ms >= result.p95_latency_ms
        assert result.min_latency_ms >= 0
        assert result.max_latency_ms >= result.min_latency_ms

    def test_exp02_threshold_validation(self):
        """EXP-02 should validate against latency thresholds."""
        from fractalstat.fractalstat_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=50)
        exp.scales = [1000]

        results, success = exp.run()

        # Success should be based on threshold comparison
        result = results[0]
        assert isinstance(result.success, bool)

    def test_exp02_get_summary(self):
        """EXP-02 should generate summary statistics."""
        from fractalstat.fractalstat_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=50)
        exp.scales = [100]

        exp.run()
        summary = exp.get_summary()

        assert isinstance(summary, dict)
        assert "total_scales_tested" in summary
        assert "all_passed" in summary
        assert "results" in summary
        assert summary["total_scales_tested"] == 1


class TestEXP02Result:
    """Tests for EXP-02 result data structure."""

    def test_exp02_result_initialization(self):
        """EXP02_Result should track retrieval metrics."""
        from fractalstat.fractalstat_experiments import EXP02_Result

        result = EXP02_Result(
            scale=1000,
            queries=500,
            mean_latency_ms=0.05,
            median_latency_ms=0.04,
            p95_latency_ms=0.08,
            p99_latency_ms=0.10,
            min_latency_ms=0.01,
            max_latency_ms=0.15,
            success=True,
        )

        assert result.scale == 1000
        assert result.queries == 500
        assert result.success

    def test_exp02_result_to_dict(self):
        """EXP02_Result should serialize to dict."""
        from fractalstat.fractalstat_experiments import EXP02_Result

        result = EXP02_Result(
            scale=1000,
            queries=500,
            mean_latency_ms=0.05,
            median_latency_ms=0.04,
            p95_latency_ms=0.08,
            p99_latency_ms=0.10,
            min_latency_ms=0.01,
            max_latency_ms=0.15,
            success=True,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["scale"] == 1000
        assert result_dict["success"]


class TestEXP03DimensionNecessity:
    """Tests for EXP-03 dimension necessity experiment."""

    def test_exp03_initialization(self):
        """EXP03_CoordinateEntropy should initialize with sample size."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=500)
        assert exp.sample_size == 500
        assert exp.results == []
        assert len(exp.FractalStat_DIMENSIONS) == 7

    def test_exp03_baseline_all_dimensions(self):
        """EXP-03 should test baseline with all 7 dimensions."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # First result should be baseline with all dimensions
        baseline = results[0]
        assert len(baseline.dimensions_used) == 7
        assert baseline.sample_size == 100

    def test_exp03_ablation_realm(self):
        """EXP-03 should test ablation when realm is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where realm was removed
        realm_ablation = [r for r in results if "realm" not in r.dimensions_used]
        assert len(realm_ablation) > 0
        assert realm_ablation[0].sample_size == 100

    def test_exp03_ablation_lineage(self):
        """EXP-03 should test ablation when lineage is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where lineage was removed
        lineage_ablation = [r for r in results if "lineage" not in r.dimensions_used]
        assert len(lineage_ablation) > 0

    def test_exp03_ablation_adjacency(self):
        """EXP-03 should test ablation when adjacency is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where adjacency was removed
        adjacency_ablation = [
            r for r in results if "adjacency" not in r.dimensions_used
        ]
        assert len(adjacency_ablation) > 0

    def test_exp03_ablation_horizon(self):
        """EXP-03 should test ablation when horizon is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where horizon was removed
        horizon_ablation = [r for r in results if "horizon" not in r.dimensions_used]
        assert len(horizon_ablation) > 0

    def test_exp03_ablation_resonance(self):
        """EXP-03 should test ablation when resonance is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where resonance was removed
        resonance_ablation = [
            r for r in results if "resonance" not in r.dimensions_used
        ]
        assert len(resonance_ablation) > 0

    def test_exp03_ablation_velocity(self):
        """EXP-03 should test ablation when velocity is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where velocity was removed
        velocity_ablation = [r for r in results if "velocity" not in r.dimensions_used]
        assert len(velocity_ablation) > 0

    def test_exp03_ablation_density(self):
        """EXP-03 should test ablation when density is removed."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        results, _ = exp.run()

        # Find result where density was removed
        density_ablation = [r for r in results if "density" not in r.dimensions_used]
        assert len(density_ablation) > 0

    def test_exp03_get_summary(self):
        """EXP-03 should generate summary statistics."""
        from fractalstat.fractalstat_experiments import EXP03_CoordinateEntropy

        exp = EXP03_CoordinateEntropy(sample_size=100)
        exp.run()
        summary = exp.get_summary()

        assert isinstance(summary, dict)
        assert "sample_size" in summary
        assert "total_dimension_combos_tested" in summary
        assert "results" in summary
        assert summary["sample_size"] == 100


class TestEXP03Result:
    """Tests for EXP-03 result data structure."""

    def test_exp03_result_initialization(self):
        """EXP03_Result should track dimension necessity metrics."""
        from fractalstat.fractalstat_experiments import EXP03_Result

        result = EXP03_Result(
            dimensions_used=["realm", "lineage", "adjacency"],
            sample_size=1000,
            shannon_entropy=2.5,
            normalized_entropy=0.8,
            entropy_reduction_pct=15.5,
            unique_coordinates=950,
            semantic_disambiguation_score=0.85,
            meets_threshold=True,
        )

        assert len(result.dimensions_used) == 3
        assert result.sample_size == 1000
        assert result.shannon_entropy == 2.5
        assert result.meets_threshold

    def test_exp03_result_to_dict(self):
        """EXP03_Result should serialize to dict."""
        from fractalstat.fractalstat_experiments import EXP03_Result

        result = EXP03_Result(
            dimensions_used=["realm", "lineage"],
            sample_size=1000,
            shannon_entropy=2.0,
            normalized_entropy=0.75,
            entropy_reduction_pct=25.0,
            unique_coordinates=750,
            semantic_disambiguation_score=0.7,
            meets_threshold=False,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["sample_size"] == 1000
        assert not result_dict["meets_threshold"]


class TestRunAllExperiments:
    """Tests for run_all_experiments orchestration."""

    def test_run_all_experiments_full(self):
        """run_all_experiments should execute all experiments."""
        from fractalstat.fractalstat_experiments import run_all_experiments

        # Use small parameters for fast testing
        results = run_all_experiments(
            exp01_samples=10,
            exp01_iterations=1,
            exp02_queries=10,
            exp03_samples=10,
        )

        assert isinstance(results, dict)
        assert "EXP-01" in results
        assert "EXP-02" in results
        assert "EXP-03" in results

    def test_run_all_experiments_selective(self):
        """run_all_experiments should handle selective experiment execution."""
        from fractalstat.fractalstat_experiments import run_all_experiments

        # Mock config to disable some experiments
        mock_config = MagicMock()
        mock_config.is_enabled.side_effect = lambda exp: exp == "EXP-01"
        mock_config.get.return_value = 10

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            results = run_all_experiments(
                exp01_samples=10,
                exp01_iterations=1,
            )

            assert "EXP-01" in results

    def test_run_all_experiments_with_config(self):
        """run_all_experiments should use config parameter overrides."""
        from fractalstat.fractalstat_experiments import run_all_experiments

        # Mock config with custom parameters
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = True
        mock_config.get.side_effect = lambda exp, param, default: 10

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            results = run_all_experiments()

            assert isinstance(results, dict)

    def test_run_all_experiments_without_config(self):
        """run_all_experiments should fallback when config unavailable."""
        from fractalstat.fractalstat_experiments import run_all_experiments

        # Mock config import to raise exception
        with patch(
            "fractalstat.config.ExperimentConfig",
            side_effect=Exception("Config not found"),
        ):
            results = run_all_experiments(
                exp01_samples=10,
                exp01_iterations=1,
                exp02_queries=10,
                exp03_samples=10,
            )

            # Should still run with default parameters
            assert isinstance(results, dict)
            assert "EXP-01" in results


class TestMainEntryPoint:
    """Tests for main entry point execution."""

    def test_main_execution(self):
        """Main entry point should execute experiments."""
        from fractalstat.fractalstat_experiments import run_all_experiments

        # Test that main execution works
        results = run_all_experiments(
            exp01_samples=5,
            exp01_iterations=1,
            exp02_queries=5,
            exp03_samples=5,
        )

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_results_json_persistence(self):
        """Main should save results to JSON file."""
        from fractalstat.fractalstat_experiments import run_all_experiments

        results = run_all_experiments(
            exp01_samples=5,
            exp01_iterations=1,
            exp02_queries=5,
            exp03_samples=5,
        )

        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, indent=2)
            temp_path = f.name

        # Verify file was created and is valid JSON
        assert Path(temp_path).exists()

        with open(temp_path, "r", encoding="UTF-8") as f:
            loaded = json.load(f)
            assert loaded == results

        # Cleanup
        Path(temp_path).unlink()


class TestEdgeCases:
    """Tests for edge cases in existing functions."""

    def test_sort_json_keys_nested(self):
        """sort_json_keys should handle deeply nested structures."""
        from fractalstat.fractalstat_experiments import sort_json_keys

        nested = {
            "z": {
                "y": {
                    "x": 1,
                    "a": 2,
                },
                "b": 3,
            },
            "a": [
                {"z": 1, "a": 2},
                {"y": 3, "b": 4},
            ],
        }

        sorted_obj = sort_json_keys(nested)

        # Check top level is sorted
        keys = list(sorted_obj.keys())
        assert keys == ["a", "z"]

        # Check nested dict is sorted
        nested_keys = list(sorted_obj["z"].keys())
        assert nested_keys == ["b", "y"]

        # Check deeply nested dict is sorted
        deep_keys = list(sorted_obj["z"]["y"].keys())
        assert deep_keys == ["a", "x"]

    def test_normalize_timestamp_timezones(self):
        """normalize_timestamp should handle different timezone formats."""
        from fractalstat.fractalstat_experiments import normalize_timestamp

        # Test with UTC timezone
        ts_utc = "2024-01-01T12:00:00Z"
        result_utc = normalize_timestamp(ts_utc)
        assert result_utc.endswith("Z")

        # Test with offset timezone
        ts_offset = "2024-01-01T12:00:00+00:00"
        result_offset = normalize_timestamp(ts_offset)
        assert result_offset.endswith("Z")

    def test_bitchain_security_fields(self):
        """BitChain should support all security fields."""
        from fractalstat.fractalstat_experiments import (
            BitChain,
            FractalStatCoordinates,
            DataClass,
        )

        coords = FractalStatCoordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            luminosity=0.5,
            polarity=Polarity.LOGIC,
            dimensionality=2,
            alignment=Alignment.TRUE_NEUTRAL,
        )

        bc = BitChain(
            id="test-security",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
            data_classification=DataClass.PII,
            access_control_list=["owner", "admin"],
            owner_id="user-123",
            encryption_key_id="key-456",
        )

        assert bc.data_classification == DataClass.PII
        assert "owner" in bc.access_control_list
        assert "admin" in bc.access_control_list
        assert bc.owner_id == "user-123"
        assert bc.encryption_key_id == "key-456"

    def test_fractalstat_uri_edge_cases(self):
        """get_fractalstat_uri should handle edge case coordinates."""
        from fractalstat.fractalstat_experiments import BitChain, FractalStatCoordinates

        # Test with extreme values
        coords = FractalStatCoordinates(
            realm="void",
            lineage=999,
            adjacency=["adj1", "adj2", "adj3"],
            horizon="crystallization",
            luminosity=-1.0,
            polarity=Polarity.CREATIVITY,
            dimensionality=3,
            alignment=Alignment.CHAOTIC_NEUTRAL,
        )

        bc = BitChain(
            id="test-uri-edge",
            entity_type="concept",
            realm="void",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        uri = bc.get_fractalstat_uri()

        assert uri.startswith("fractalstat://")
        assert "void" in uri
        assert "999" in uri
        assert "crystallization" in uri
        assert "luminosity=" in uri  # Check for new coordinate parameter names
        assert "polarity=" in uri
        assert "dimensionality=" in uri
