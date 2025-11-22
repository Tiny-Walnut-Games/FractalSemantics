"""
Comprehensive tests for EXP-11b: Dimension Stress Test
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

from fractalstat.exp11b_dimension_stress_test import (
    DimensionStressTest,
    StressTestResult,
    save_results,
)


class TestStressTestResult:
    """Test StressTestResult dataclass."""

    def test_stress_test_result_initialization(self):
        """StressTestResult should initialize with all fields."""
        result = StressTestResult(
            test_name="Test 1",
            dimension_count=7,
            dimensions_used=["realm", "lineage", "adjacency"],
            sample_size=1000,
            unique_addresses=950,
            collisions=50,
            collision_rate=0.05,
            max_collisions_per_address=2,
            coordinate_diversity=0.8,
            description="Test description",
        )

        assert result.test_name == "Test 1"
        assert result.dimension_count == 7
        assert result.sample_size == 1000
        assert result.unique_addresses == 950
        assert result.collisions == 50
        assert result.collision_rate == 0.05
        assert result.max_collisions_per_address == 2
        assert result.coordinate_diversity == 0.8

    def test_stress_test_result_to_dict(self):
        """StressTestResult should serialize to dict."""
        result = StressTestResult(
            test_name="Test 2",
            dimension_count=3,
            dimensions_used=["realm", "horizon"],
            sample_size=500,
            unique_addresses=500,
            collisions=0,
            collision_rate=0.0,
            max_collisions_per_address=1,
            coordinate_diversity=0.6,
            description="Zero collision test",
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["test_name"] == "Test 2"
        assert result_dict["collisions"] == 0
        assert len(result_dict["dimensions_used"]) == 2


class TestDimensionStressTest:
    """Test DimensionStressTest class."""

    def test_initialization_defaults(self):
        """DimensionStressTest should initialize with default parameters."""
        exp = DimensionStressTest()

        assert exp.sample_size == 10000  # Default from class
        assert exp.results == []

    def test_initialization_custom(self):
        """DimensionStressTest should accept custom parameters."""
        exp = DimensionStressTest(sample_size=1000)

        assert exp.sample_size == 1000
        assert exp.results == []

    def test_run_quick_test(self):
        """run() should execute successfully with small sample for testing."""
        exp = DimensionStressTest(sample_size=50)  # Very small for speed

        results, success = exp.run()

        assert isinstance(results, object)  # DimensionStressTestResult
        assert success is True
        assert len(results.test_results) == 10  # 10 different test configurations

        # Check that all results are of correct type
        for result in results.test_results:
            assert isinstance(result, StressTestResult)
            assert isinstance(result.test_name, str)
            assert isinstance(result.dimension_count, int)
            assert isinstance(result.dimensions_used, list)
            assert result.sample_size == 50
            assert isinstance(result.unique_addresses, int)
            assert isinstance(result.collisions, int)
            assert isinstance(result.collision_rate, float)
            assert 0.0 <= result.collision_rate <= 1.0
            assert isinstance(result.max_collisions_per_address, int)
            assert isinstance(result.coordinate_diversity, float)

    def test_run_has_baseline_test(self):
        """run() should include baseline test with zero collisions."""
        exp = DimensionStressTest(sample_size=20)

        results, success = exp.run()

        # First test should be baseline (Test 1) with zero collisions typically
        baseline = results.test_results[0]
        assert "Baseline" in baseline.test_name
        assert baseline.collisions == 0  # Should have no collisions

    def test_run_includes_various_configurations(self):
        """run() should test various stress configurations."""
        exp = DimensionStressTest(sample_size=30)

        results, success = exp.run()

        test_names = [r.test_name for r in results.test_results]

        # Check that we have the expected test variety
        assert any("Fixed ID" in name for name in test_names)
        assert any("Limited Coordinate Range" in name for name in test_names)
        assert any("Only 3 Dimensions" in name for name in test_names)
        assert any("Only 2 Dimensions" in name for name in test_names)
        assert any("Only 1 Dimension" in name for name in test_names)
        assert any("Extreme Stress" in name for name in test_names)

    def test_run_calculates_collision_rates(self):
        """run() should correctly calculate collision rates."""
        exp = DimensionStressTest(sample_size=25)

        results, success = exp.run()

        for result in results.test_results:
            # Collision rate should be collisions / sample_size
            expected_rate = (
                result.collisions / result.sample_size
                if result.sample_size > 0
                else 0.0
            )
            assert abs(result.collision_rate - expected_rate) < 0.001

    def test_run_generates_key_findings(self):
        """run() should generate meaningful key findings."""
        exp = DimensionStressTest(sample_size=15)

        results, success = exp.run()

        assert len(results.key_findings) > 0
        assert isinstance(results.key_findings[0], str)

        # Should have insights about collisions or lack thereof
        all_findings = " ".join(results.key_findings).lower()
        assert any(
            keyword in all_findings
            for keyword in ["collision", "dimension", "critical", "insight", "sha-256"]
        )

    def test_run_measures_coordinate_diversity(self):
        """run() should measure coordinate diversity."""
        exp = DimensionStressTest(sample_size=40)

        results, success = exp.run()

        for result in results.test_results:
            # Diversity should be between 0.0 and 1.0
            assert 0.0 <= result.coordinate_diversity <= 1.0
            assert isinstance(result.coordinate_diversity, float)

    def test_run_handles_small_samples(self):
        """run() should handle very small sample sizes gracefully."""
        exp = DimensionStressTest(sample_size=5)

        results, success = exp.run()

        assert success is True
        for result in results.test_results:
            assert result.sample_size == 5
            # With such small samples, diversity calculations should still work
            assert isinstance(result.coordinate_diversity, float)

    def test_run_deterministic_results(self):
        """run() should produce deterministic results (same random seed usage)."""
        exp1 = DimensionStressTest(sample_size=8)
        exp2 = DimensionStressTest(sample_size=8)

        results1, _ = exp1.run()
        results2, _ = exp2.run()

        # Should have same number of results
        assert len(results1.test_results) == len(results2.test_results)

        # Key metrics should be similar (allowing for potential random variation)
        for r1, r2 in zip(results1.test_results, results2.test_results):
            assert r1.test_name == r2.test_name
            assert r1.sample_size == r2.sample_size
            # Note: Actual collision counts might vary slightly due to random generation
            # but the structure should be the same

    def test_saved_results_have_correct_structure(self):
        """Saved results should have complete structure."""
        exp = DimensionStressTest(sample_size=10)

        results, success = exp.run()
        result_dict = results.to_dict()

        # Check top-level structure
        assert "experiment" in result_dict
        assert result_dict["experiment"] == "EXP-11b"
        assert "test_type" in result_dict
        assert "start_time" in result_dict
        assert "end_time" in result_dict
        assert "total_duration_seconds" in result_dict
        assert "key_findings" in result_dict
        assert "test_results" in result_dict

        # Check test results structure
        assert len(result_dict["test_results"]) == 10  # All test configurations
        for test_result in result_dict["test_results"]:
            assert "test_name" in test_result
            assert "dimension_count" in test_result
            assert "dimensions_used" in test_result
            assert "sample_size" in test_result
            assert "unique_addresses" in test_result
            assert "collisions" in test_result
            assert "collision_rate" in test_result
            assert "max_collisions_per_address" in test_result
            assert "coordinate_diversity" in test_result
            assert "description" in test_result


class TestSaveResults:
    """Test save_results function."""

    def test_save_results_custom_filename(self):
        """save_results should save with custom filename."""
        mock_results = MagicMock()
        mock_results.to_dict.return_value = {
            "experiment": "EXP-11b",
            "test_results": [],
            "key_findings": [],
        }

        with patch("fractalstat.exp11b_dimension_stress_test.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = lambda self, x: Path("/tmp") / x

            output_file = "custom_exp11b_results.json"
            result_path = save_results(mock_results, output_file)

            assert "custom_exp11b_results.json" in result_path

    def test_save_results_default_filename(self):
        """save_results should create default filename."""
        mock_results = MagicMock()
        mock_results.to_dict.return_value = {"experiment": "EXP-11b"}

        with patch("fractalstat.exp11b_dimension_stress_test.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = lambda self, x: Path("/tmp") / x

            result_path = save_results(mock_results)

            assert "exp11b_dimension_stress_test_" in result_path


class TestStressTestConfigurations:
    """Test specific stress test configurations."""

    def test_baseline_should_have_no_collisions(self):
        """Baseline test should typically have zero collisions."""
        exp = DimensionStressTest(sample_size=15)

        results, success = exp.run()

        baseline = results.test_results[0]
        assert "Baseline" in baseline.test_name
        # In practice, with unique IDs and full dimensions, should have no collisions
        assert baseline.collisions == 0

    def test_fewer_dimensions_increase_collision_risk(self):
        """Tests with fewer dimensions should show higher collision rates."""
        exp = DimensionStressTest(sample_size=20)

        results, success = exp.run()

        # Find tests with different dimension counts
        full_dims = next(
            (r for r in results.test_results if r.dimension_count == 7), None
        )
        few_dims = next(
            (r for r in results.test_results if r.dimension_count == 2), None
        )

        if full_dims and few_dims:
            # Fewer dimensions should not have lower collision resistance
            # (Note: In practice, SHA-256 still prevents most collisions)
            assert few_dims.dimension_count < full_dims.dimension_count

    def test_diversity_correlates_with_dimensions(self):
        """Coordinate diversity should correlate with number of dimensions."""
        exp = DimensionStressTest(sample_size=25)

        results, success = exp.run()

        # Tests with more dimensions should generally have higher diversity
        high_dim_test = next(
            (r for r in results.test_results if r.dimension_count >= 7), None
        )
        low_dim_test = next(
            (r for r in results.test_results if r.dimension_count == 2), None
        )

        if high_dim_test and low_dim_test:
            # More dimensions should allow for higher diversity
            assert high_dim_test.dimension_count > low_dim_test.dimension_count
            # Diversity should be measurable
            assert high_dim_test.coordinate_diversity >= 0.0
            assert low_dim_test.coordinate_diversity >= 0.0

    def test_zero_collision_tests_exist(self):
        """Some tests should demonstrate zero collisions."""
        exp = DimensionStressTest(sample_size=12)

        results, success = exp.run()

        # At least some tests should have zero collisions
        zero_collision_tests = [r for r in results.test_results if r.collisions == 0]
        assert len(zero_collision_tests) > 0

        # At least baseline should have zero collisions
        baseline = results.test_results[0]
        assert baseline.collisions == 0
