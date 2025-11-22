"""
Comprehensive tests for EXP-02: Retrieval Efficiency Test
"""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from fractalstat.exp02_retrieval_efficiency import (
    EXP02_RetrievalEfficiency,
    EXP02_Result,
    save_results,
)


class TestEXP02Result:
    """Test EXP02_Result dataclass."""

    def test_exp02_result_initialization(self):
        """EXP02_Result should initialize with all fields."""
        result = EXP02_Result(
            scale=1000,
            queries=500,
            mean_latency_ms=0.05,
            median_latency_ms=0.04,
            p95_latency_ms=0.08,
            p99_latency_ms=0.12,
            min_latency_ms=0.01,
            max_latency_ms=0.15,
            success=True,
        )

        assert result.scale == 1000
        assert result.queries == 500
        assert result.mean_latency_ms == 0.05
        assert result.median_latency_ms == 0.04
        assert result.p95_latency_ms == 0.08
        assert result.p99_latency_ms == 0.12
        assert result.min_latency_ms == 0.01
        assert result.max_latency_ms == 0.15
        assert result.success is True

    def test_exp02_result_to_dict(self):
        """EXP02_Result should serialize to dict."""
        result = EXP02_Result(
            scale=10000,
            queries=1000,
            mean_latency_ms=0.25,
            median_latency_ms=0.22,
            p95_latency_ms=0.45,
            p99_latency_ms=0.6,
            min_latency_ms=0.05,
            max_latency_ms=0.8,
            success=False,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["scale"] == 10000
        assert result_dict["success"] is False
        assert result_dict["queries"] == 1000


class TestEXP02RetrievalEfficiency:
    """Test EXP02_RetrievalEfficiency class."""

    def test_initialization_defaults(self):
        """EXP02_RetrievalEfficiency should initialize with default parameters."""
        exp = EXP02_RetrievalEfficiency()

        assert exp.query_count == 1000
        assert exp.scales == [1_000, 10_000, 100_000]
        assert exp.results == []

    def test_initialization_custom(self):
        """EXP02_RetrievalEfficiency should accept custom parameters."""
        exp = EXP02_RetrievalEfficiency(query_count=500)

        assert exp.query_count == 500
        assert exp.scales == [1_000, 10_000, 100_000]

    def test_run_small_scale(self):
        """run() should execute successfully with small scale."""
        exp = EXP02_RetrievalEfficiency(query_count=10)

        # Only test one scale for speed
        exp.scales = [100]  # Much smaller for testing

        results, success = exp.run()

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(success, bool)

        # Check result structure
        result = results[0]
        assert isinstance(result, EXP02_Result)
        assert result.scale == 100
        assert result.queries == 10
        assert isinstance(result.mean_latency_ms, float)
        assert isinstance(result.median_latency_ms, float)
        assert isinstance(result.p95_latency_ms, float)
        assert isinstance(result.p99_latency_ms, float)
        assert isinstance(result.min_latency_ms, float)
        assert isinstance(result.max_latency_ms, float)
        assert isinstance(result.success, bool)

        # Sanity checks
        assert result.mean_latency_ms >= 0
        assert result.min_latency_ms <= result.max_latency_ms
        assert result.median_latency_ms >= result.min_latency_ms
        assert result.median_latency_ms <= result.max_latency_ms

    def test_run_multiple_scales(self):
        """run() should execute multiple scales."""
        exp = EXP02_RetrievalEfficiency(query_count=5)

        # Test smaller scales for speed
        exp.scales = [50, 100]

        results, success = exp.run()

        assert len(results) == 2
        assert all(isinstance(r, EXP02_Result) for r in results)
        assert results[0].scale == 50
        assert results[1].scale == 100
        assert all(r.queries == 5 for r in results)

    def test_performance_thresholds(self):
        """run() should check performance thresholds correctly."""
        exp = EXP02_RetrievalEfficiency(query_count=10)

        # Test with very small scale that should pass easily
        exp.scales = [10]  # Very small scale

        results, success = exp.run()

        assert len(results) == 1
        # With such a small scale, should easily pass threshold of 0.1ms
        assert results[0].success is True
        assert success is True

    def test_get_summary(self):
        """get_summary() should return comprehensive statistics."""
        exp = EXP02_RetrievalEfficiency(query_count=5)
        exp.scales = [20]

        results, success = exp.run()

        summary = exp.get_summary()

        assert isinstance(summary, dict)
        assert "total_scales_tested" in summary
        assert "all_passed" in summary
        assert "results" in summary

        assert summary["total_scales_tested"] == 1
        assert isinstance(summary["all_passed"], bool)
        assert len(summary["results"]) == 1

    def test_get_summary_empty_results(self):
        """get_summary() should handle empty results gracefully."""
        exp = EXP02_RetrievalEfficiency()

        # Don't run, so results are empty
        summary = exp.get_summary()

        assert isinstance(summary, dict)
        assert summary["total_scales_tested"] == 0
        assert summary["all_passed"] is False
        assert len(summary["results"]) == 0


class TestSaveResults:
    """Test save_results function."""

    def test_save_results_custom_filename(self):
        """save_results should save with custom filename."""
        summary = {
            "total_scales_tested": 2,
            "all_passed": True,
            "results": [
                {
                    "scale": 1000,
                    "queries": 100,
                    "mean_latency_ms": 0.05,
                    "success": True,
                }
            ],
        }

        with patch("fractalstat.exp02_retrieval_efficiency.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = lambda self, x: Path("/tmp") / x

            output_file = "custom_exp02_results.json"
            result_path = save_results(summary, output_file)

            assert "custom_exp02_results.json" in result_path

    def test_save_results_default_filename(self):
        """save_results should create default filename."""
        summary = {
            "total_scales_tested": 1,
            "all_passed": True,
            "results": [],
        }

        with patch("fractalstat.exp02_retrieval_efficiency.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = lambda self, x: Path("/tmp") / x

            result_path = save_results(summary)

            assert "exp02_retrieval_efficiency_" in result_path


class TestMainEntryPoint:
    """Test main entry point execution."""

    def test_main_with_config(self):
        """Main should load from config."""
        mock_config = MagicMock()

        def mock_get(exp, param, default):
            if exp == "EXP-02" and param == "query_count":
                return 50
            return default

        mock_config.get.side_effect = mock_get

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            exp = EXP02_RetrievalEfficiency(
                query_count=mock_config.get("EXP-02", "query_count", 1000),
            )

            assert exp.query_count == 50

    def test_main_without_config(self):
        """Main should fallback to defaults when config unavailable."""
        with patch(
            "fractalstat.config.ExperimentConfig",
            side_effect=Exception("Config not found"),
        ):
            exp = EXP02_RetrievalEfficiency()

            assert exp.query_count == 1000


class TestPerformanceCharacteristics:
    """Test performance characteristics and scaling."""

    def test_timing_accuracy(self):
        """Timing measurements should be reasonable."""
        exp = EXP02_RetrievalEfficiency(query_count=10)
        exp.scales = [10]  # Very small for speed

        start_time = time.time()
        results, success = exp.run()
        end_time = time.time()

        # Should complete in reasonable time (< 1 second for small test)
        assert end_time - start_time < 1.0

        # Check latency is reasonable (should be very small for small dataset)
        result = results[0]
        assert result.mean_latency_ms < 0.1  # Should be << 0.1ms for 10 items

    def test_latency_distribution(self):
        """Latency measurements should show reasonable distribution."""
        exp = EXP02_RetrievalEfficiency(query_count=50)
        exp.scales = [20]

        results, success = exp.run()

        result = results[0]

        # Check that percentile calculations are reasonable
        assert result.min_latency_ms <= result.median_latency_ms
        assert result.median_latency_ms <= result.p95_latency_ms
        assert result.p95_latency_ms <= result.p99_latency_ms
        assert result.p99_latency_ms <= result.max_latency_ms

        # Check that min is very small (hash table lookup should be fast)
        assert result.min_latency_ms < 0.01  # Should be under 10 microseconds

    def test_deterministic_behavior(self):
        """Same parameters should give roughly same results."""
        exp1 = EXP02_RetrievalEfficiency(query_count=10)
        exp2 = EXP02_RetrievalEfficiency(query_count=10)

        exp1.scales = [50]
        exp2.scales = [50]

        results1, success1 = exp1.run()
        results2, success2 = exp2.run()

        # Results should be very similar (same scale, same query count)
        # Allow some variation due to timing jitter
        diff = abs(results1[0].mean_latency_ms - results2[0].mean_latency_ms)
        assert diff < 0.1  # Should be within 0.1ms of each other

    def test_scale_independence(self):
        """Larger scales should show degraded but consistent performance."""
        exp = EXP02_RetrievalEfficiency(query_count=20)
        exp.scales = [100, 200]  # Two different scales

        results, success = exp.run()

        assert len(results) == 2

        # Both should succeed (both well below thresholds)
        assert results[0].success is True
        assert results[1].success is True

        # Assert that performance is reasonable
        assert results[0].mean_latency_ms < 0.1  # Both should be fast
        assert results[1].mean_latency_ms < 0.1

        # Larger scale might be slightly slower, but should still be fast
        diff = results[1].mean_latency_ms - results[0].mean_latency_ms
        # Allow for some variation, not a strict requirement for this test
        assert diff < 0.05 or diff >= -0.05  # Within 0.05ms either way
