"""
Test suite for EXP-03: Coordinate Space Entropy Test

Tests entropy calculation, normalization, semantic disambiguation metrics,
and ablation testing for STAT7 dimensions.
"""

import numpy as np


class TestEXP03Result:
    """Test EXP03_Result data structure."""

    def test_exp03_result_initialization(self):
        """EXP03_Result should track entropy metrics."""
        from fractalstat.exp03_coordinate_entropy import EXP03_Result

        result = EXP03_Result(
            dimensions_used=["realm", "lineage", "adjacency"],
            sample_size=1000,
            shannon_entropy=8.5,
            normalized_entropy=0.85,
            entropy_reduction_pct=15.0,
            unique_coordinates=950,
            semantic_disambiguation_score=0.92,
            meets_threshold=True,
        )

        assert len(result.dimensions_used) == 3
        assert result.sample_size == 1000
        assert result.shannon_entropy == 8.5
        assert result.normalized_entropy == 0.85
        assert result.entropy_reduction_pct == 15.0
        assert result.unique_coordinates == 950
        assert result.semantic_disambiguation_score == 0.92
        assert result.meets_threshold

    def test_exp03_result_to_dict(self):
        """EXP03_Result should serialize to dict."""
        from fractalstat.exp03_coordinate_entropy import EXP03_Result

        result = EXP03_Result(
            dimensions_used=["realm", "lineage"],
            sample_size=1000,
            shannon_entropy=7.2,
            normalized_entropy=0.72,
            entropy_reduction_pct=3.5,
            unique_coordinates=900,
            semantic_disambiguation_score=0.88,
            meets_threshold=False,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["sample_size"] == 1000
        assert result_dict["shannon_entropy"] == 7.2
        assert not result_dict["meets_threshold"]


class TestEXP03CoordinateEntropy:
    """Test EXP03_CoordinateEntropy experiment class."""

    def test_exp03_initialization(self):
        """EXP03_CoordinateEntropy should initialize with sample size and seed."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=500, random_seed=123)
        assert exp.sample_size == 500
        assert exp.random_seed == 123
        assert exp.results == []
        assert exp.baseline_entropy is None
        assert len(exp.STAT7_DIMENSIONS) == 7

    def test_exp03_dimensions_defined(self):
        """EXP03 should have all 7 STAT7 dimensions defined."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()
        assert "realm" in exp.STAT7_DIMENSIONS
        assert "lineage" in exp.STAT7_DIMENSIONS
        assert "adjacency" in exp.STAT7_DIMENSIONS
        assert "horizon" in exp.STAT7_DIMENSIONS
        assert "resonance" in exp.STAT7_DIMENSIONS
        assert "velocity" in exp.STAT7_DIMENSIONS
        assert "density" in exp.STAT7_DIMENSIONS


class TestShannonEntropy:
    """Test Shannon entropy calculation."""

    def test_compute_shannon_entropy_uniform(self):
        """Shannon entropy should be maximum for uniform distribution."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # Uniform distribution: all unique values
        coords = [f"coord_{i}" for i in range(100)]
        entropy = exp.compute_shannon_entropy(coords)

        # Maximum entropy for 100 unique values is log2(100) ≈ 6.644
        expected_max = np.log2(100)
        assert abs(entropy - expected_max) < 0.001

    def test_compute_shannon_entropy_single_value(self):
        """Shannon entropy should be zero for single repeated value."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # All same value
        coords = ["same"] * 100
        entropy = exp.compute_shannon_entropy(coords)

        assert entropy == 0.0

    def test_compute_shannon_entropy_empty(self):
        """Shannon entropy should handle empty list."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()
        entropy = exp.compute_shannon_entropy([])
        assert entropy == 0.0

    def test_compute_shannon_entropy_two_values(self):
        """Shannon entropy should be 1.0 for equal split of two values."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # 50/50 split
        coords = ["A"] * 50 + ["B"] * 50
        entropy = exp.compute_shannon_entropy(coords)

        # Entropy of 50/50 split is 1.0 bit
        assert abs(entropy - 1.0) < 0.001

    def test_compute_shannon_entropy_skewed(self):
        """Shannon entropy should be lower for skewed distribution."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # Skewed: 90% A, 10% B
        coords = ["A"] * 90 + ["B"] * 10
        entropy = exp.compute_shannon_entropy(coords)

        # Should be less than 1.0 (the maximum for 2 values)
        assert 0 < entropy < 1.0


class TestEntropyNormalization:
    """Test entropy normalization."""

    def test_normalize_entropy_maximum(self):
        """Normalized entropy should be 1.0 for maximum entropy."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # Maximum entropy for 100 samples
        max_entropy = np.log2(100)
        normalized = exp.normalize_entropy(max_entropy, 100)

        assert abs(normalized - 1.0) < 0.001

    def test_normalize_entropy_zero(self):
        """Normalized entropy should be 0.0 for zero entropy."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()
        normalized = exp.normalize_entropy(0.0, 100)
        assert normalized == 0.0

    def test_normalize_entropy_half(self):
        """Normalized entropy should be 0.5 for half maximum."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        max_entropy = np.log2(100)
        half_entropy = max_entropy / 2
        normalized = exp.normalize_entropy(half_entropy, 100)

        assert abs(normalized - 0.5) < 0.001

    def test_normalize_entropy_edge_cases(self):
        """Normalized entropy should handle edge cases."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # Single sample
        assert exp.normalize_entropy(0.0, 1) == 0.0

        # Zero samples
        assert exp.normalize_entropy(0.0, 0) == 0.0


class TestSemanticDisambiguation:
    """Test semantic disambiguation score calculation."""

    def test_semantic_disambiguation_perfect(self):
        """Disambiguation score should be high for all unique coordinates."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # All unique
        coords = [f"coord_{i}" for i in range(100)]
        score = exp.compute_semantic_disambiguation_score(coords, 100)

        # Should be close to 1.0
        assert score > 0.9

    def test_semantic_disambiguation_poor(self):
        """Disambiguation score should be low for many duplicates."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # All same
        coords = ["same"] * 100
        score = exp.compute_semantic_disambiguation_score(coords, 1)

        # Should be close to 0.0
        assert score < 0.5

    def test_semantic_disambiguation_empty(self):
        """Disambiguation score should handle empty list."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()
        score = exp.compute_semantic_disambiguation_score([], 0)
        assert score == 0.0

    def test_semantic_disambiguation_partial(self):
        """Disambiguation score should be moderate for partial uniqueness."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # 50% unique
        coords = [f"coord_{i}" for i in range(50)] + ["dup"] * 50
        score = exp.compute_semantic_disambiguation_score(coords, 51)

        # Should be between 0 and 1
        assert 0 < score < 1


class TestCoordinateExtraction:
    """Test coordinate extraction from bit-chains."""

    def test_extract_coordinates_all_dimensions(self):
        """Should extract coordinates with all dimensions."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )
        from fractalstat.stat7_experiments import generate_random_bitchain

        exp = EXP03_CoordinateEntropy()
        bitchains = [generate_random_bitchain(seed=i) for i in range(10)]

        coords = exp.extract_coordinates(bitchains, exp.STAT7_DIMENSIONS)

        assert len(coords) == 10
        assert all(isinstance(c, str) for c in coords)

    def test_extract_coordinates_subset(self):
        """Should extract coordinates with dimension subset."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )
        from fractalstat.stat7_experiments import generate_random_bitchain

        exp = EXP03_CoordinateEntropy()
        bitchains = [generate_random_bitchain(seed=i) for i in range(10)]

        # Extract only realm and lineage
        coords = exp.extract_coordinates(bitchains, ["realm", "lineage"])

        assert len(coords) == 10
        # Each coordinate should only contain realm and lineage
        for coord_str in coords:
            import json

            coord_dict = json.loads(coord_str)
            assert "realm" in coord_dict
            assert "lineage" in coord_dict
            assert "adjacency" not in coord_dict

    def test_extract_coordinates_deterministic(self):
        """Coordinate extraction should be deterministic."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )
        from fractalstat.stat7_experiments import generate_random_bitchain

        exp = EXP03_CoordinateEntropy()
        bitchains = [generate_random_bitchain(seed=i) for i in range(10)]

        coords1 = exp.extract_coordinates(bitchains, exp.STAT7_DIMENSIONS)
        coords2 = exp.extract_coordinates(bitchains, exp.STAT7_DIMENSIONS)

        assert coords1 == coords2


class TestEXP03Run:
    """Test EXP-03 experiment execution."""

    def test_exp03_run_small_sample(self):
        """EXP-03 should run with small sample size."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results, success = exp.run()

        assert isinstance(results, list)
        assert len(results) == 8  # Baseline + 7 ablations
        assert isinstance(success, bool)

    def test_exp03_baseline_result(self):
        """EXP-03 should have baseline result with all dimensions."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results, _ = exp.run()

        baseline = results[0]
        assert len(baseline.dimensions_used) == 7
        assert baseline.entropy_reduction_pct == 0.0
        assert baseline.meets_threshold

    def test_exp03_ablation_results(self):
        """EXP-03 should have ablation results for each dimension."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results, _ = exp.run()

        # Should have 7 ablation results
        ablations = results[1:]
        assert len(ablations) == 7

        # Each should have 6 dimensions (one removed)
        for result in ablations:
            assert len(result.dimensions_used) == 6

    def test_exp03_entropy_reduction_calculated(self):
        """EXP-03 should calculate entropy reduction for ablations."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results, _ = exp.run()

        baseline = results[0]

        for ablation in results[1:]:
            # Entropy reduction should be non-negative
            assert ablation.entropy_reduction_pct >= 0

            # Ablation entropy should be <= baseline entropy
            assert ablation.shannon_entropy <= baseline.shannon_entropy

    def test_exp03_baseline_entropy_stored(self):
        """EXP-03 should store baseline entropy."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        exp.run()

        assert exp.baseline_entropy is not None
        assert exp.baseline_entropy > 0

    def test_exp03_reproducible_with_seed(self):
        """EXP-03 should be reproducible with same seed."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp1 = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results1, _ = exp1.run()

        exp2 = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results2, _ = exp2.run()

        # Should have same baseline entropy
        assert abs(results1[0].shannon_entropy - results2[0].shannon_entropy) < 0.001


class TestEXP03Summary:
    """Test EXP-03 summary generation."""

    def test_exp03_get_summary(self):
        """EXP-03 should generate summary statistics."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        exp.run()
        summary = exp.get_summary()

        assert isinstance(summary, dict)
        assert "sample_size" in summary
        assert "random_seed" in summary
        assert "baseline_entropy" in summary
        assert "total_tests" in summary
        assert "all_critical" in summary
        assert "results" in summary

    def test_exp03_summary_values(self):
        """EXP-03 summary should have correct values."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        exp.run()
        summary = exp.get_summary()

        assert summary["sample_size"] == 50
        assert summary["random_seed"] == 42
        assert summary["total_tests"] == 8
        assert isinstance(summary["all_critical"], bool)


class TestVisualizationData:
    """Test visualization data generation."""

    def test_generate_visualization_data(self):
        """Should generate visualization data."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        exp.run()
        viz_data = exp.generate_visualization_data()

        assert isinstance(viz_data, dict)
        assert "dimensions" in viz_data
        assert "entropy_reductions" in viz_data
        assert "baseline_entropy" in viz_data
        assert "threshold" in viz_data

    def test_visualization_data_structure(self):
        """Visualization data should have correct structure."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        exp.run()
        viz_data = exp.generate_visualization_data()

        assert len(viz_data["dimensions"]) == 7
        assert len(viz_data["entropy_reductions"]) == 7
        assert viz_data["threshold"] == 5.0

    def test_visualization_data_empty_before_run(self):
        """Visualization data should be empty before run."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        viz_data = exp.generate_visualization_data()

        assert viz_data == {}


class TestSaveResults:
    """Test results saving functionality."""

    def test_save_results_creates_file(self):
        """save_results should create JSON file."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
            save_results,
        )
        import tempfile
        import json
        from pathlib import Path

        exp = EXP03_CoordinateEntropy(sample_size=10, random_seed=42)
        exp.run()
        summary = exp.get_summary()

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_results.json"

            # Temporarily override results directory
            import fractalstat.exp03_coordinate_entropy as exp03_module

            original_file = exp03_module.__file__
            exp03_module.__file__ = str(Path(tmpdir) / "exp03_coordinate_entropy.py")

            try:
                save_results(summary, str(output_file.name))

                # Check file exists
                results_dir = Path(tmpdir) / "results"
                saved_file = results_dir / output_file.name
                assert saved_file.exists()

                # Check valid JSON
                with open(saved_file) as f:
                    loaded = json.load(f)
                    assert loaded == summary
            finally:
                exp03_module.__file__ = original_file


class TestPlotEntropyContributions:
    """Test entropy contribution plotting."""

    def test_plot_without_matplotlib(self):
        """plot_entropy_contributions should handle missing matplotlib."""
        from fractalstat.exp03_coordinate_entropy import (
            plot_entropy_contributions,
        )

        viz_data = {
            "dimensions": ["realm", "lineage"],
            "entropy_reductions": [10.0, 5.0],
            "baseline_entropy": 8.0,
            "threshold": 5.0,
        }

        # Should not raise error even if matplotlib is missing
        plot_entropy_contributions(viz_data)

    def test_plot_with_empty_data(self):
        """plot_entropy_contributions should handle empty data."""
        from fractalstat.exp03_coordinate_entropy import (
            plot_entropy_contributions,
        )

        # Should not raise error with empty data
        plot_entropy_contributions({})


class TestMainEntryPoint:
    """Test main entry point execution."""

    def test_main_with_config(self):
        """Main should use config if available."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )
        from unittest.mock import patch, MagicMock

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda exp, param, default: {
            ("EXP-03", "sample_size", 1000): 50,
            ("EXP-03", "random_seed", 42): 123,
        }.get((exp, param, default), default)

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=123)
            results, _ = exp.run()

            assert len(results) == 8

    def test_main_without_config(self):
        """Main should use defaults if config unavailable."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=50, random_seed=42)
        results, _ = exp.run()

        assert len(results) == 8


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_exp03_with_zero_samples(self):
        """EXP-03 should handle zero samples gracefully."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=0, random_seed=42)
        # Should not crash
        try:
            exp.run()
        except Exception:
            # Expected to fail with 0 samples
            pass

    def test_exp03_with_single_sample(self):
        """EXP-03 should handle single sample."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy(sample_size=1, random_seed=42)
        results, _ = exp.run()

        # Should complete but with low entropy
        assert len(results) == 8

    def test_shannon_entropy_with_duplicates(self):
        """Shannon entropy should handle duplicate coordinates correctly."""
        from fractalstat.exp03_coordinate_entropy import (
            EXP03_CoordinateEntropy,
        )

        exp = EXP03_CoordinateEntropy()

        # Mix of unique and duplicate
        coords = ["A", "A", "B", "B", "C"]
        entropy = exp.compute_shannon_entropy(coords)

        # Should be between 0 and log2(5)
        assert 0 < entropy < np.log2(5)
