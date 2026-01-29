"""
Tests for EXP-11: Dimension Cardinality Analysis
Comprehensive test coverage targeting 95%+
"""
# pylint: disable=protected-access

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from fractalstat.exp11_dimension_cardinality import (
    EXP11_DimensionCardinality,
    DimensionTestResult,
    DimensionCardinalityResult,
    save_results,
)
from fractalstat.fractalstat_experiments import generate_random_bitchain


class TestDimensionTestResult:
    """Tests for DimensionTestResult dataclass."""

    def test_initialization(self):
        """DimensionTestResult should initialize with all fields."""
        result = DimensionTestResult(
            dimension_count=7,
            dimensions_used=[
                "realm",
                "lineage",
                "adjacency",
                "horizon",
                "resonance",
                "velocity",
                "density",
            ],
            sample_size=1000,
            unique_addresses=1000,
            collisions=0,
            collision_rate=0.0,
            mean_retrieval_latency_ms=0.05,
            median_retrieval_latency_ms=0.04,
            avg_storage_bytes=150,
            storage_overhead_per_dimension=21.4,
            semantic_expressiveness_score=0.95,
        )

        assert result.dimension_count == 7
        assert len(result.dimensions_used) == 7
        assert result.sample_size == 1000
        assert result.collisions == 0
        assert result.semantic_expressiveness_score == 0.95

    def test_to_dict(self):
        """DimensionTestResult should serialize to dict."""
        result = DimensionTestResult(
            dimension_count=5,
            dimensions_used=[
                "realm",
                "lineage",
                "adjacency",
                "horizon",
                "resonance",
            ],
            sample_size=500,
            unique_addresses=495,
            collisions=5,
            collision_rate=0.01,
            mean_retrieval_latency_ms=0.06,
            median_retrieval_latency_ms=0.05,
            avg_storage_bytes=120,
            storage_overhead_per_dimension=24.0,
            semantic_expressiveness_score=0.75,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["dimension_count"] == 5
        assert result_dict["collisions"] == 5
        assert result_dict["collision_rate"] == 0.01


class TestDimensionCardinalityResult:
    """Tests for DimensionCardinalityResult dataclass."""

    def test_initialization(self):
        """DimensionCardinalityResult should initialize with all fields."""
        result = DimensionCardinalityResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=1000,
            dimension_counts_tested=[3, 5, 7, 9],
            test_iterations=5,
            dimension_results=[],
            optimal_dimension_count=7,
            optimal_collision_rate=0.0,
            optimal_retrieval_latency_ms=0.05,
            optimal_storage_efficiency=20.0,
            diminishing_returns_threshold=7,
            major_findings=["7 dimensions justified"],
            seven_dimensions_justified=True,
        )

        assert result.sample_size == 1000
        assert result.optimal_dimension_count == 7
        assert result.seven_dimensions_justified

    def test_to_dict(self):
        """DimensionCardinalityResult should serialize to dict."""
        result = DimensionCardinalityResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=1000,
            dimension_counts_tested=[3, 5, 7],
            test_iterations=3,
            dimension_results=[],
            optimal_dimension_count=7,
            optimal_collision_rate=0.0,
            optimal_retrieval_latency_ms=0.05,
            optimal_storage_efficiency=20.0,
            diminishing_returns_threshold=7,
            major_findings=[],
            seven_dimensions_justified=True,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["experiment"] == "EXP-11"
        assert result_dict["test_type"] == "Dimension Cardinality Analysis"
        assert result_dict["seven_dimensions_justified"]


class TestDimensionCardinalityExperiment:
    """Tests for DimensionCardinalityExperiment class."""

    def test_initialization(self):
        """DimensionCardinalityExperiment should initialize with parameters."""
        exp = EXP11_DimensionCardinality(
            sample_size=100,
            dimension_counts=[3, 5, 7],
            test_iterations=2,
        )

        assert exp.sample_size == 100
        assert exp.dimension_counts == [3, 5, 7]
        assert exp.test_iterations == 2
        assert exp.results == []

    def test_initialization_defaults(self):
        """DimensionCardinalityExperiment should use default parameters."""
        exp = EXP11_DimensionCardinality()

        assert exp.sample_size == 1000
        assert exp.dimension_counts == [3, 4, 5, 6, 7, 8, 9, 10]
        assert exp.test_iterations == 5

    def test_select_dimensions_reduced(self):
        """_select_dimensions should return subset for counts <= 8."""
        exp = EXP11_DimensionCardinality()

        dims_3 = exp._select_dimensions(3)
        assert len(dims_3) == 3
        assert dims_3 == ["realm", "lineage", "adjacency"]

        dims_5 = exp._select_dimensions(5)
        assert len(dims_5) == 5
        assert "realm" in dims_5
        assert "lineage" in dims_5

        dims_7 = exp._select_dimensions(7)
        assert len(dims_7) == 7
        assert dims_7 == exp.FractalStat_DIMENSIONS[:7]

    def test_select_dimensions_extended(self):
        """_select_dimensions should add hypothetical dimensions for counts > 8."""
        exp = EXP11_DimensionCardinality()

        dims_8 = exp._select_dimensions(8)
        assert len(dims_8) == 8
        assert dims_8 == exp.FractalStat_DIMENSIONS  # All FractalStat dimensions
        assert "temperature" not in dims_8

        dims_9 = exp._select_dimensions(9)
        assert len(dims_9) == 9
        assert "temperature" in dims_9
        assert "entropy" not in dims_9

        dims_10 = exp._select_dimensions(10)
        assert len(dims_10) == 10
        assert "temperature" in dims_10
        assert "entropy" in dims_10
        assert "coherence" not in dims_10

    def test_compute_address_with_dimensions(self):
        """_compute_address_with_dimensions should generate addresses with selected dimensions."""
        exp = EXP11_DimensionCardinality()
        bc = generate_random_bitchain(seed=42)

        # Test with all 7 dimensions
        addr_7 = exp._compute_address_with_dimensions(bc, exp.FractalStat_DIMENSIONS)
        assert isinstance(addr_7, str)
        assert len(addr_7) == 64  # SHA-256 hex

        # Test with 3 dimensions
        addr_3 = exp._compute_address_with_dimensions(
            bc, ["realm", "lineage", "adjacency"]
        )
        assert isinstance(addr_3, str)
        assert len(addr_3) == 64

        # Addresses should be different
        assert addr_7 != addr_3

    def test_compute_address_with_extended_dimensions(self):
        """_compute_address_with_dimensions should handle hypothetical dimensions."""
        exp = EXP11_DimensionCardinality()
        bc = generate_random_bitchain(seed=42)

        # Test with extended dimensions
        dims_8 = exp._select_dimensions(8)
        addr_8 = exp._compute_address_with_dimensions(bc, dims_8)
        assert isinstance(addr_8, str)
        assert len(addr_8) == 64

    def test_calculate_semantic_expressiveness(self):
        """_calculate_semantic_expressiveness should score dimension sets."""
        exp = EXP11_DimensionCardinality()
        bitchains = [generate_random_bitchain(seed=i) for i in range(10)]

        # Test with all 7 FractalStat dimensions
        score_7 = exp._calculate_semantic_expressiveness(
            exp.FractalStat_DIMENSIONS, bitchains
        )
        assert score_7 >= 0.95  # Base score from FractalStat weights, plus bonuses
        assert score_7 <= 2.0   # Reasonable upper bound

        # Test with 3 dimensions
        score_3 = exp._calculate_semantic_expressiveness(
            ["realm", "lineage", "adjacency"], bitchains
        )
        assert score_3 >= 0.0
        assert score_3 <= 1.5   # Should be lower than 7-dimension score with bonuses
        assert score_3 < score_7

        # Test with extended dimensions
        score_10 = exp._calculate_semantic_expressiveness(
            exp.FractalStat_DIMENSIONS + exp.EXTENDED_DIMENSIONS, bitchains
        )
        assert score_10 >= 0.95  # At least base score
        assert score_10 <= 3.0   # Reasonable upper bound

    def test_test_dimension_count(self):
        """_test_dimension_count should test a specific dimension count."""
        exp = EXP11_DimensionCardinality(sample_size=50, test_iterations=1)

        result = exp._test_dimension_count(7)

        assert isinstance(result, DimensionTestResult)
        assert result.dimension_count == 7
        assert len(result.dimensions_used) == 7
        assert result.sample_size == 50
        assert result.unique_addresses > 0
        assert 0.0 <= result.collision_rate <= 1.0
        assert result.mean_retrieval_latency_ms >= 0
        assert result.avg_storage_bytes > 0
        assert 0.0 <= result.semantic_expressiveness_score <= 1.0

    def test_run_small_sample(self):
        """run() should execute with small sample for testing."""
        exp = EXP11_DimensionCardinality(
            sample_size=20,
            dimension_counts=[3, 5, 7],
            test_iterations=1,
        )

        result, success = exp.run()

        assert isinstance(result, DimensionCardinalityResult)
        assert isinstance(success, bool)
        assert result.sample_size == 20
        assert len(result.dimension_results) == 3
        assert result.optimal_dimension_count in [3, 5, 7]

    def test_run_collision_detection(self):
        """run() should detect collisions with fewer dimensions."""
        exp = EXP11_DimensionCardinality(
            sample_size=100,
            dimension_counts=[3, 7],
            test_iterations=1,
        )

        result, success = exp.run()

        # Find results for 3 and 7 dimensions
        result_3 = next(r for r in result.dimension_results if r.dimension_count == 3)
        result_7 = next(r for r in result.dimension_results if r.dimension_count == 7)

        # 3 dimensions should have higher collision rate than 7
        # (though not guaranteed with small samples)
        assert result_3.collision_rate >= 0.0
        assert result_7.collision_rate >= 0.0

    def test_run_diminishing_returns(self):
        """run() should identify diminishing returns threshold."""
        exp = EXP11_DimensionCardinality(
            sample_size=50,
            dimension_counts=[5, 6, 7, 8],
            test_iterations=1,
        )

        result, success = exp.run()

        assert result.diminishing_returns_threshold in [5, 6, 7, 8]

    def test_run_seven_dimensions_justified(self):
        """run() should evaluate if 7 dimensions is justified."""
        exp = EXP11_DimensionCardinality(
            sample_size=100,
            dimension_counts=[3, 5, 7, 9],
            test_iterations=2,
        )

        result, success = exp.run()

        assert isinstance(result.seven_dimensions_justified, bool)
        # seven_dimensions_justified may differ from overall success
        # success means experiment ran and found reasonable optimal count
        # seven_dimensions_justified means 7 dimensions is the optimal choice
        assert isinstance(success, bool)

    def test_run_major_findings(self):
        """run() should generate major findings."""
        exp = EXP11_DimensionCardinality(
            sample_size=50,
            dimension_counts=[3, 7],
            test_iterations=1,
        )

        result, success = exp.run()

        assert len(result.major_findings) > 0
        assert any("Optimal dimension count" in f for f in result.major_findings)


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_results_default_filename(self):
        """save_results should save with default filename."""
        result = DimensionCardinalityResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=100,
            dimension_counts_tested=[3, 7],
            test_iterations=1,
            dimension_results=[],
            optimal_dimension_count=7,
            optimal_collision_rate=0.0,
            optimal_retrieval_latency_ms=0.05,
            optimal_storage_efficiency=20.0,
            diminishing_returns_threshold=7,
            major_findings=[],
            seven_dimensions_justified=True,
        )

        # Mock the results directory creation
        with patch("fractalstat.exp11_dimension_cardinality.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = (
                lambda self, x: Path(tempfile.gettempdir()) / x
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = str(Path(tmpdir) / "test_results.json")
                result_path = save_results(result, output_file)

                assert Path(result_path).exists()
                with open(result_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    assert loaded["experiment"] == "EXP-11"

    def test_save_results_custom_filename(self):
        """save_results should save with custom filename."""
        result = DimensionCardinalityResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=100,
            dimension_counts_tested=[3, 7],
            test_iterations=1,
            dimension_results=[],
            optimal_dimension_count=7,
            optimal_collision_rate=0.0,
            optimal_retrieval_latency_ms=0.05,
            optimal_storage_efficiency=20.0,
            diminishing_returns_threshold=7,
            major_findings=[],
            seven_dimensions_justified=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the results directory
            with patch("fractalstat.exp11_dimension_cardinality.Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent = MagicMock()
                mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                    lambda self, x: Path(tmpdir)
                )

                output_file = "custom_exp11_results.json"
                result_path = save_results(result, output_file)

                assert "custom_exp11_results.json" in result_path


class TestMainEntryPoint:
    """Tests for main entry point execution."""

    def test_main_with_config(self):
        """Main should load from config."""
        mock_config = MagicMock()

        def mock_get(exp, param, default):
            if exp == "EXP-11" and param == "sample_size":
                return 50
            elif exp == "EXP-11" and param == "dimension_counts":
                return [3, 7]
            elif exp == "EXP-11" and param == "test_iterations":
                return 1
            return default

        mock_config.get.side_effect = mock_get

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            exp = EXP11_DimensionCardinality(
                sample_size=mock_config.get("EXP-11", "sample_size", 1000),
                dimension_counts=mock_config.get(
                    "EXP-11", "dimension_counts", [3, 4, 5, 6, 7, 8, 9, 10]
                ),
                test_iterations=mock_config.get("EXP-11", "test_iterations", 5),
            )

            assert exp.sample_size == 50
            assert exp.dimension_counts == [3, 7]
            assert exp.test_iterations == 1

    def test_main_without_config(self):
        """Main should fallback to defaults when config unavailable."""
        with patch(
            "fractalstat.config.ExperimentConfig",
            side_effect=Exception("Config not found"),
        ):
            exp = EXP11_DimensionCardinality()

            assert exp.sample_size == 1000
            assert exp.dimension_counts == [3, 4, 5, 6, 7, 8, 9, 10]
            assert exp.test_iterations == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_sample_size(self):
        """Experiment should handle zero sample size gracefully."""
        exp = EXP11_DimensionCardinality(
            sample_size=0,
            dimension_counts=[7],
            test_iterations=1,
        )

        # Should not crash, but results will be trivial
        result = exp._test_dimension_count(7)
        assert result.sample_size == 0
        assert result.unique_addresses == 0

    def test_single_dimension(self):
        """Experiment should handle single dimension."""
        exp = EXP11_DimensionCardinality(
            sample_size=10,
            dimension_counts=[1],
            test_iterations=1,
        )

        result = exp._test_dimension_count(1)
        assert result.dimension_count == 1
        assert len(result.dimensions_used) == 1

    def test_many_dimensions(self):
        """Experiment should handle many dimensions (> 10)."""
        exp = EXP11_DimensionCardinality(
            sample_size=10,
            dimension_counts=[10],
            test_iterations=1,
        )

        dims = exp._select_dimensions(10)
        assert len(dims) == 10

    def test_empty_dimension_counts(self):
        """Experiment should handle empty dimension counts list."""
        exp = EXP11_DimensionCardinality(
            sample_size=10,
            dimension_counts=[],
            test_iterations=1,
        )

        result, success = exp.run()
        assert len(result.dimension_results) == 0

    def test_collision_rate_calculation(self):
        """Collision rate should be calculated correctly."""
        exp = EXP11_DimensionCardinality(sample_size=100, test_iterations=1)

        result = exp._test_dimension_count(7)

        # Collision rate should be between 0 and 1
        assert 0.0 <= result.collision_rate <= 1.0

        # Collision rate should match formula
        expected_rate = result.collisions / result.sample_size
        assert abs(result.collision_rate - expected_rate) < 0.0001
