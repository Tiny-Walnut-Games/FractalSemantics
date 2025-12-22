"""
Test suite for EXP-04: Bit-Chain FractalStat Fractal Scaling Test
Tests address collision detection and retrieval performance at scale.
"""


class TestScaleTestConfig:
    """Test ScaleTestConfig for fractal scaling."""

    def test_scale_config_initializes(self):
        """ScaleTestConfig should initialize with scale parameters."""
        from fractalstat.exp04_fractal_scaling import ScaleTestConfig

        config = ScaleTestConfig(scale=1000, num_retrievals=100, timeout_seconds=60)
        assert config.scale == 1000
        assert config.num_retrievals == 100
        assert config.timeout_seconds == 60

    def test_scale_name_formatting(self):
        """Scale name should format large numbers correctly."""
        from fractalstat.exp04_fractal_scaling import ScaleTestConfig

        config_1k = ScaleTestConfig(scale=1_000, num_retrievals=100, timeout_seconds=60)
        assert config_1k.name() == "1K"

        config_100k = ScaleTestConfig(
            scale=100_000, num_retrievals=100, timeout_seconds=60
        )
        assert config_100k.name() == "100K"

        config_1m = ScaleTestConfig(
            scale=1_000_000, num_retrievals=100, timeout_seconds=60
        )
        assert config_1m.name() == "1M"


class TestScaleTestResults:
    """Test ScaleTestResults data structure."""

    def test_results_initialization(self):
        """ScaleTestResults should initialize with all fields."""
        from fractalstat.exp04_fractal_scaling import ScaleTestResults

        results = ScaleTestResults(
            scale=1000,
            num_bitchains=1000,
            num_addresses=1000,
            unique_addresses=1000,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=100,
            retrieval_times_ms=[0.1] * 100,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=200.0,
        )

        assert results.scale == 1000
        assert results.collision_count == 0

    def test_results_validity_check(self):
        """is_valid should correctly identify valid results."""
        from fractalstat.exp04_fractal_scaling import ScaleTestResults

        valid_results = ScaleTestResults(
            scale=1000,
            num_bitchains=1000,
            num_addresses=1000,
            unique_addresses=1000,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=100,
            retrieval_times_ms=[0.1] * 100,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=200.0,
        )

        assert valid_results.is_valid()

    def test_results_to_dict(self):
        """Results should convert to serializable dict."""
        from fractalstat.exp04_fractal_scaling import ScaleTestResults

        results = ScaleTestResults(
            scale=1000,
            num_bitchains=1000,
            num_addresses=1000,
            unique_addresses=1000,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=100,
            retrieval_times_ms=[0.1] * 100,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=200.0,
        )

        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert "scale" in result_dict
        assert "collision_rate_percent" in result_dict
        assert "retrieval" in result_dict


class TestFractalScalingResults:
    """Test FractalScalingResults aggregation."""

    def test_results_initialization(self):
        """FractalScalingResults should aggregate scale results."""
        from fractalstat.exp04_fractal_scaling import (
            FractalScalingResults,
            ScaleTestResults,
        )

        scale_result = ScaleTestResults(
            scale=1000,
            num_bitchains=1000,
            num_addresses=1000,
            unique_addresses=1000,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=100,
            retrieval_times_ms=[0.1] * 100,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=200.0,
        )

        results = FractalScalingResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            scale_results=[scale_result],
            collision_degradation="✓ Zero collisions",
            retrieval_degradation="✓ Logarithmic scaling",
            is_fractal=True,
        )

        assert results.is_fractal
        assert len(results.scale_results) == 1

    def test_results_to_dict(self):
        """Results should serialize to dict."""
        from fractalstat.exp04_fractal_scaling import (
            FractalScalingResults,
            ScaleTestResults,
        )

        scale_result = ScaleTestResults(
            scale=1000,
            num_bitchains=1000,
            num_addresses=1000,
            unique_addresses=1000,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=100,
            retrieval_times_ms=[0.1] * 100,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=200.0,
        )

        results = FractalScalingResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            scale_results=[scale_result],
            collision_degradation="✓ Zero collisions",
            retrieval_degradation="✓ Logarithmic scaling",
            is_fractal=True,
        )

        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["experiment"] == "EXP-04"
        assert "scale_results" in result_dict


class TestDegradationAnalysis:
    """Test degradation analysis function."""

    def test_analyze_degradation_zero_collisions(self):
        """analyze_degradation should detect zero collision state."""
        from fractalstat.exp04_fractal_scaling import (
            analyze_degradation,
            ScaleTestResults,
        )

        scale_results = [
            ScaleTestResults(
                scale=1000,
                num_bitchains=1000,
                num_addresses=1000,
                unique_addresses=1000,
                collision_count=0,
                collision_rate=0.0,
                num_retrievals=100,
                retrieval_times_ms=[0.1] * 100,
                retrieval_mean_ms=0.1,
                retrieval_median_ms=0.1,
                retrieval_p95_ms=0.15,
                retrieval_p99_ms=0.2,
                total_time_seconds=5.0,
                addresses_per_second=200.0,
            ),
            ScaleTestResults(
                scale=10000,
                num_bitchains=10000,
                num_addresses=10000,
                unique_addresses=10000,
                collision_count=0,
                collision_rate=0.0,
                num_retrievals=100,
                retrieval_times_ms=[0.15] * 100,
                retrieval_mean_ms=0.15,
                retrieval_median_ms=0.15,
                retrieval_p95_ms=0.2,
                retrieval_p99_ms=0.25,
                total_time_seconds=50.0,
                addresses_per_second=200.0,
            ),
        ]

        collision_msg, retrieval_msg, is_fractal = analyze_degradation(scale_results)

        assert "Zero collisions" in collision_msg
        assert is_fractal


class TestSaveResults:
    """Test results persistence."""

    def test_save_results_returns_filename(self):
        """save_results should return output file path."""
        from fractalstat.exp04_fractal_scaling import (
            save_results,
            FractalScalingResults,
            ScaleTestResults,
        )
        import tempfile
        import os

        scale_result = ScaleTestResults(
            scale=1000,
            num_bitchains=1000,
            num_addresses=1000,
            unique_addresses=1000,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=100,
            retrieval_times_ms=[0.1] * 100,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=200.0,
        )

        results = FractalScalingResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            scale_results=[scale_result],
            collision_degradation="✓ Zero collisions",
            retrieval_degradation="✓ Logarithmic scaling",
            is_fractal=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_exp04_results.json")
            result_path = save_results(results, output_file)
            assert os.path.exists(result_path)
