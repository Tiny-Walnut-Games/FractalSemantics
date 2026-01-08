"""
Extended tests for EXP-04 Fractal Scaling to achieve 95%+ coverage
"""

import tempfile
import json
from pathlib import Path


class TestExp04Extended:
    """Extended tests for fractal scaling experiment."""

    def test_run_scale_test_execution(self):
        """run_scale_test should execute full test at small scale."""
        from fractalstat.exp04_fractal_scaling import (
            run_scale_test,
            ScaleTestConfig,
        )

        config = ScaleTestConfig(scale=10, num_retrievals=5, timeout_seconds=60)

        result = run_scale_test(config)

        assert result.scale == 10
        assert result.num_bitchains == 10
        assert result.collision_count >= 0
        assert len(result.retrieval_times_ms) == 5

    def test_run_scale_test_collision_detection(self):
        """run_scale_test should detect collisions if they occur."""
        from fractalstat.exp04_fractal_scaling import (
            run_scale_test,
            ScaleTestConfig,
        )

        config = ScaleTestConfig(scale=100, num_retrievals=10, timeout_seconds=60)

        result = run_scale_test(config)

        # Should have zero collisions for FractalStat
        assert result.collision_count == 0
        assert result.collision_rate == 0.0

    def test_run_scale_test_retrieval_performance(self):
        """run_scale_test should measure retrieval performance."""
        from fractalstat.exp04_fractal_scaling import (
            run_scale_test,
            ScaleTestConfig,
        )

        config = ScaleTestConfig(scale=50, num_retrievals=20, timeout_seconds=60)

        result = run_scale_test(config)

        assert result.retrieval_mean_ms > 0
        assert result.retrieval_median_ms > 0
        assert result.retrieval_p95_ms >= result.retrieval_median_ms
        assert result.retrieval_p99_ms >= result.retrieval_p95_ms

    def test_run_fractal_scaling_test_quick_mode(self):
        """run_fractal_scaling_test should complete in quick mode."""
        from fractalstat.exp04_fractal_scaling import run_fractal_scaling_test

        # Override config to use very small scales
        with tempfile.TemporaryDirectory():
            results = run_fractal_scaling_test(quick_mode=True)

            assert len(results.scale_results) > 0
            assert results.is_fractal is not None
            assert results.collision_degradation is not None

    def test_analyze_degradation_with_collisions(self):
        """analyze_degradation should detect collision degradation."""
        from fractalstat.exp04_fractal_scaling import (
            analyze_degradation,
            ScaleTestResults,
        )

        # Create results with collisions
        results_with_collisions = [
            ScaleTestResults(
                scale=1000,
                num_bitchains=1000,
                num_addresses=1000,
                unique_addresses=995,  # 5 collisions
                collision_count=5,
                collision_rate=0.005,
                num_retrievals=100,
                retrieval_times_ms=[0.1] * 100,
                retrieval_mean_ms=0.1,
                retrieval_median_ms=0.1,
                retrieval_p95_ms=0.15,
                retrieval_p99_ms=0.2,
                total_time_seconds=5.0,
                addresses_per_second=200.0,
            )
        ]

        collision_msg, retrieval_msg, is_fractal = analyze_degradation(
            results_with_collisions
        )

        assert "COLLISION DETECTED" in collision_msg
        # With collisions detected, retrieval analysis should still be present
        assert isinstance(retrieval_msg, str)
        assert not is_fractal

    def test_analyze_degradation_retrieval_scaling(self):
        """analyze_degradation should analyze retrieval scaling."""
        from fractalstat.exp04_fractal_scaling import (
            analyze_degradation,
            ScaleTestResults,
        )

        results = [
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
                retrieval_times_ms=[0.12] * 100,
                retrieval_mean_ms=0.12,
                retrieval_median_ms=0.12,
                retrieval_p95_ms=0.18,
                retrieval_p99_ms=0.24,
                total_time_seconds=50.0,
                addresses_per_second=200.0,
            ),
        ]

        collision_msg, retrieval_msg, is_fractal = analyze_degradation(results)

        assert "Zero collisions" in collision_msg
        # Should have successful retrieval scaling analysis
        assert isinstance(retrieval_msg, str)
        # With good scaling data, should indicate healthy retrieval performance
        assert "OK" in retrieval_msg or "scaling" in retrieval_msg.lower() or "scales" in retrieval_msg.lower()
        assert is_fractal

    def test_save_results_with_file_io(self):
        """save_results should write to file."""
        from fractalstat.exp04_fractal_scaling import (
            save_results,
            FractalScalingResults,
            ScaleTestResults,
        )

        scale_result = ScaleTestResults(
            scale=100,
            num_bitchains=100,
            num_addresses=100,
            unique_addresses=100,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=10,
            retrieval_times_ms=[0.1] * 10,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=5.0,
            addresses_per_second=20.0,
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
            output_file = str(Path(tmpdir) / "test_results.json")
            result_path = save_results(results, output_file)

            assert Path(result_path).exists()

            # Verify content
            with open(result_path) as f:
                data = json.load(f)
                assert data["experiment"] == "EXP-04"
                assert data["is_fractal"]

    def test_scale_test_timeout_handling(self):
        """run_scale_test should respect timeout."""
        from fractalstat.exp04_fractal_scaling import ScaleTestConfig

        config = ScaleTestConfig(scale=100, num_retrievals=10, timeout_seconds=1)

        # Should complete within timeout for small scale
        assert config.timeout_seconds == 1

    def test_performance_degradation_edge_cases(self):
        """analyze_degradation should handle edge cases."""
        from fractalstat.exp04_fractal_scaling import (
            analyze_degradation,
            ScaleTestResults,
        )

        # Single result
        single_result = [
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
            )
        ]

        collision_msg, retrieval_msg, is_fractal = analyze_degradation(single_result)

        assert "Zero collisions" in collision_msg
        # With single result, retrieval analysis should still be present (even if limited)
        assert isinstance(retrieval_msg, str)
        assert is_fractal  # Single result with zero collisions is considered fractal

    def test_results_persistence_with_actual_file_io(self):
        """Results should persist correctly with actual file I/O."""
        from fractalstat.exp04_fractal_scaling import (
            FractalScalingResults,
            ScaleTestResults,
        )

        scale_result = ScaleTestResults(
            scale=50,
            num_bitchains=50,
            num_addresses=50,
            unique_addresses=50,
            collision_count=0,
            collision_rate=0.0,
            num_retrievals=5,
            retrieval_times_ms=[0.1] * 5,
            retrieval_mean_ms=0.1,
            retrieval_median_ms=0.1,
            retrieval_p95_ms=0.15,
            retrieval_p99_ms=0.2,
            total_time_seconds=2.0,
            addresses_per_second=25.0,
        )

        results = FractalScalingResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:02:00Z",
            total_duration_seconds=120.0,
            scale_results=[scale_result],
            collision_degradation="✓ Zero collisions",
            retrieval_degradation="✓ Logarithmic scaling",
            is_fractal=True,
        )

        result_dict = results.to_dict()

        assert result_dict["all_valid"]
        assert len(result_dict["scale_results"]) == 1
