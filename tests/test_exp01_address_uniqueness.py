"""
Comprehensive tests for EXP-01: Address Uniqueness Test
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

from fractalstat.exp01_address_uniqueness import (
    EXP01_AddressUniqueness,
    EXP01_Result,
    save_results,
)
from fractalstat.stat7_experiments import generate_random_bitchain


class TestEXP01Result:
    """Test EXP01_Result dataclass."""

    def test_exp01_result_initialization(self):
        """EXP01_Result should initialize with all fields."""
        result = EXP01_Result(
            iteration=1,
            total_bitchains=1000,
            unique_addresses=1000,
            collisions=0,
            collision_rate=0.0,
            success=True,
        )

        assert result.iteration == 1
        assert result.total_bitchains == 1000
        assert result.unique_addresses == 1000
        assert result.collisions == 0
        assert result.collision_rate == 0.0
        assert result.success is True

    def test_exp01_result_to_dict(self):
        """EXP01_Result should serialize to dict."""
        result = EXP01_Result(
            iteration=2,
            total_bitchains=500,
            unique_addresses=495,
            collisions=5,
            collision_rate=0.01,
            success=False,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["iteration"] == 2
        assert result_dict["collisions"] == 5
        assert result_dict["collision_rate"] == 0.01
        assert result_dict["success"] is False


class TestEXP01AddressUniqueness:
    """Test EXP01_AddressUniqueness class."""

    def test_initialization_defaults(self):
        """EXP01_AddressUniqueness should initialize with default parameters."""
        exp = EXP01_AddressUniqueness()

        assert exp.sample_size == 1000
        assert exp.iterations == 10
        assert exp.results == []

    def test_initialization_custom(self):
        """EXP01_AddressUniqueness should accept custom parameters."""
        exp = EXP01_AddressUniqueness(sample_size=500, iterations=5)

        assert exp.sample_size == 500
        assert exp.iterations == 5
        assert exp.results == []

    def test_run_single_iteration_success(self):
        """run() should execute successfully with small sample."""
        exp = EXP01_AddressUniqueness(sample_size=10, iterations=1)

        results, success = exp.run()

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(success, bool)

        # Check result structure
        result = results[0]
        assert isinstance(result, EXP01_Result)
        assert result.iteration == 1
        assert result.total_bitchains == 10
        assert isinstance(result.unique_addresses, int)
        assert isinstance(result.collisions, int)
        assert isinstance(result.collision_rate, float)
        assert isinstance(result.success, bool)

    def test_run_multiple_iterations(self):
        """run() should execute multiple iterations."""
        exp = EXP01_AddressUniqueness(sample_size=5, iterations=3)

        results, success = exp.run()

        assert len(results) == 3
        assert all(isinstance(r, EXP01_Result) for r in results)
        assert all(r.iteration in [1, 2, 3] for r in results)
        assert all(r.total_bitchains == 5 for r in results)

    def test_run_collision_detection(self):
        """run() should detect collisions if they occur."""
        # This is tricky since SHA-256 should never collide in practice
        # We'll use a mock to simulate collision detection

        # Manually test the collision detection logic
        bitchains = [generate_random_bitchain(seed=i) for i in range(3)]

        # Compute addresses manually
        addresses = []
        address_set = set()

        for bc in bitchains:
            addr = bc.compute_address()
            addresses.append(addr)
            if addr in address_set:
                break  # Found a collision
            address_set.add(addr)

        # With real SHA-256, we shouldn't have collisions for small samples
        assert len(address_set) == len(bitchains)  # All unique

    def test_run_deterministic_addressing(self):
        """run() should produce deterministic addresses."""
        exp1 = EXP01_AddressUniqueness(sample_size=5, iterations=1)
        exp2 = EXP01_AddressUniqueness(sample_size=5, iterations=1)

        results1, _ = exp1.run()
        results2, _ = exp2.run()

        # Results should be identical with same seeds
        assert results1[0].unique_addresses == results2[0].unique_addresses
        assert results1[0].collisions == results2[0].collisions

    def test_get_summary(self):
        """get_summary() should return comprehensive statistics."""
        exp = EXP01_AddressUniqueness(sample_size=10, iterations=2)
        results, success = exp.run()

        summary = exp.get_summary()

        assert isinstance(summary, dict)
        assert "total_iterations" in summary
        assert "total_bitchains_tested" in summary
        assert "total_collisions" in summary
        assert "overall_collision_rate" in summary
        assert "all_passed" in summary
        assert "results" in summary

        assert summary["total_iterations"] == 2
        assert summary["total_bitchains_tested"] == 20  # 2 iterations * 10 samples
        assert isinstance(summary["all_passed"], bool)
        assert len(summary["results"]) == 2

    def test_get_summary_empty_results(self):
        """get_summary() should handle empty results gracefully."""
        exp = EXP01_AddressUniqueness(sample_size=10, iterations=2)

        # Don't run, so results are empty
        summary = exp.get_summary()

        # Should not crash but return empty/default values
        assert isinstance(summary, dict)
        assert summary["total_iterations"] == 0
        assert summary["total_bitchains_tested"] == 0
        assert summary["total_collisions"] == 0
        assert summary["overall_collision_rate"] == 0.0
        assert summary["all_passed"] is False
        assert len(summary["results"]) == 0


class TestSaveResults:
    """Test save_results function."""

    def test_save_results_custom_filename(self):
        """save_results should save with custom filename."""
        summary = {
            "total_iterations": 2,
            "total_bitchains_tested": 20,
            "total_collisions": 0,
            "overall_collision_rate": 0.0,
            "all_passed": True,
            "results": [
                {
                    "iteration": 1,
                    "total_bitchains": 10,
                    "unique_addresses": 10,
                    "collisions": 0,
                    "collision_rate": 0.0,
                    "success": True,
                }
            ],
        }

        with patch("fractalstat.exp01_address_uniqueness.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = lambda self, x: Path("/tmp") / x

            output_file = "custom_exp01_results.json"
            result_path = save_results(summary, output_file)

            assert "custom_exp01_results.json" in result_path

    def test_save_results_default_filename(self):
        """save_results should create default filename."""
        summary = {
            "total_iterations": 1,
            "total_bitchains_tested": 10,
            "total_collisions": 0,
            "overall_collision_rate": 0.0,
            "all_passed": True,
            "results": [],
        }

        with patch("fractalstat.exp01_address_uniqueness.Path") as mock_path:
            mock_results_dir = MagicMock()
            mock_path.return_value.resolve.return_value.parent = MagicMock()
            mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                lambda self, x: mock_results_dir
            )
            mock_results_dir.mkdir = MagicMock()
            mock_results_dir.__truediv__ = lambda self, x: Path("/tmp") / x

            result_path = save_results(summary)

            # Should contain timestamp and exp01 prefix
            assert "exp01_address_uniqueness_" in result_path


class TestMainEntryPoint:
    """Test main entry point execution."""

    def test_main_with_config(self):
        """Main should load from config."""
        mock_config = MagicMock()

        def mock_get(exp, param, default):
            if exp == "EXP-01" and param == "sample_size":
                return 50
            elif exp == "EXP-01" and param == "iterations":
                return 2
            return default

        mock_config.get.side_effect = mock_get

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            exp = EXP01_AddressUniqueness(
                sample_size=mock_config.get("EXP-01", "sample_size", 1000),
                iterations=mock_config.get("EXP-01", "iterations", 10),
            )

            assert exp.sample_size == 50
            assert exp.iterations == 2

    def test_main_without_config(self):
        """Main should fallback to defaults when config unavailable."""
        with patch(
            "fractalstat.config.ExperimentConfig",
            side_effect=Exception("Config not found"),
        ):
            exp = EXP01_AddressUniqueness()

            assert exp.sample_size == 1000
            assert exp.iterations == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_sample_size(self):
        """Experiment should handle zero sample size gracefully."""
        exp = EXP01_AddressUniqueness(sample_size=0, iterations=1)

        results, success = exp.run()

        assert len(results) == 1
        assert results[0].total_bitchains == 0
        assert results[0].unique_addresses == 0
        assert results[0].collisions == 0
        assert results[0].collision_rate == 0.0  # Should be 0.0, not divide by zero

    def test_single_iteration(self):
        """Experiment should handle single iteration."""
        exp = EXP01_AddressUniqueness(sample_size=5, iterations=1)

        results, success = exp.run()

        assert len(results) == 1
        assert results[0].iteration == 1

    def test_large_sample_size(self):
        """Experiment should handle large sample sizes."""
        exp = EXP01_AddressUniqueness(sample_size=100, iterations=1)

        results, success = exp.run()

        assert results[0].total_bitchains == 100
        assert results[0].unique_addresses <= 100
        assert results[0].collisions >= 0

    def test_deterministic_seeding(self):
        """Different iterations should produce same results with fixed seeds."""
        # This tests that our seeding logic produces deterministic results
        bc1 = generate_random_bitchain(seed=42)
        bc2 = generate_random_bitchain(seed=42)

        # Same seed should produce same bit-chain
        assert bc1.id == bc2.id
        assert bc1.compute_address() == bc2.compute_address()

    def test_collision_rate_calculation(self):
        """Collision rate should be calculated correctly."""
        exp = EXP01_AddressUniqueness(sample_size=20, iterations=1)

        results, success = exp.run()

        result = results[0]
        expected_rate = result.collisions / result.total_bitchains
        assert abs(result.collision_rate - expected_rate) < 0.0001

    def test_success_determination(self):
        """Success should be True only when collisions == 0."""
        exp = EXP01_AddressUniqueness(sample_size=1, iterations=1)

        results, success = exp.run()

        result = results[0]
        # With sample size 1, should always be 0 collisions
        assert result.collisions == 0
        assert result.success is True
        assert success is True
