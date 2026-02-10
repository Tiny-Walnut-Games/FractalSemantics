"""
Extended tests for EXP-06 Entanglement Detection to achieve 95%+ coverage
"""

import pytest


class TestExp06Extended:
    """Extended tests for entanglement detection."""

    def test_entanglement_detector_with_large_dataset(self):
        """EntanglementDetector should handle large datasets."""
        from fractalsemantics.exp06_entanglement_detection import (
            EntanglementDetector,
        )

        # Create many bitchains
        bitchains = []
        for i in range(20):
            bitchains.append(
                {
                    "id": f"bc{i}",
                    "coordinates": {
                        "realm": "data" if i % 2 == 0 else "narrative",
                        "lineage": i,
                        "adjacency": [f"n{j}" for j in range(i % 3)],
                        "horizon": "genesis",
                        "resonance": 0.5,
                        "velocity": 0.5,
                        "density": 0.5,
                    },
                }
            )

        detector = EntanglementDetector(threshold=0.8)
        entangled = detector.detect(bitchains)

        assert isinstance(entangled, list)
        assert len(detector.scores) > 0

    def test_validation_result_edge_cases(self):
        """compute_validation_metrics should handle edge cases."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_validation_metrics,
        )

        # All true positives
        true_pairs = {("a", "b"), ("c", "d")}
        detected_pairs = {("a", "b"), ("c", "d")}

        result = compute_validation_metrics(true_pairs, detected_pairs, 10)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0

    def test_validation_result_all_false_positives(self):
        """compute_validation_metrics should handle all false positives."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_validation_metrics,
        )

        true_pairs = {("a", "b")}
        detected_pairs = {("c", "d"), ("e", "f")}

        result = compute_validation_metrics(true_pairs, detected_pairs, 10)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0

    def test_entanglement_detector_get_score_distribution(self):
        """get_score_distribution should return statistics."""
        from fractalsemantics.exp06_entanglement_detection import (
            EntanglementDetector,
        )

        bitchains = [
            {
                "id": f"bc{i}",
                "coordinates": {
                    "realm": "data",
                    "lineage": i,
                    "adjacency": [],
                    "horizon": "genesis",
                    "resonance": 0.5,
                    "velocity": 0.5,
                    "density": 0.5,
                },
            }
            for i in range(5)
        ]

        detector = EntanglementDetector()
        detector.detect(bitchains)

        dist = detector.get_score_distribution()

        assert "min" in dist
        assert "max" in dist
        assert "mean" in dist
        assert "median" in dist
        assert "std_dev" in dist

    def test_entanglement_detector_get_all_scores(self):
        """get_all_scores should return all computed scores."""
        from fractalsemantics.exp06_entanglement_detection import (
            EntanglementDetector,
        )

        bitchains = [
            {
                "id": f"bc{i}",
                "coordinates": {
                    "realm": "data",
                    "lineage": i,
                    "adjacency": [],
                    "horizon": "genesis",
                    "resonance": 0.5,
                    "velocity": 0.5,
                    "density": 0.5,
                },
            }
            for i in range(3)
        ]

        detector = EntanglementDetector()
        detector.detect(bitchains)

        all_scores = detector.get_all_scores()

        assert len(all_scores) == 3  # 3 choose 2 = 3 pairs
        assert all(isinstance(s, dict) for s in all_scores)

    def test_entanglement_detector_score_method(self):
        """score method should compute single pair score."""
        from fractalsemantics.exp06_entanglement_detection import (
            EntanglementDetector,
        )

        bc1 = {
            "id": "bc1",
            "coordinates": {
                "realm": "data",
                "lineage": 1,
                "adjacency": [],
                "horizon": "genesis",
                "resonance": 0.5,
                "velocity": 0.5,
                "density": 0.5,
            },
        }

        bc2 = {
            "id": "bc2",
            "coordinates": {
                "realm": "data",
                "lineage": 2,
                "adjacency": [],
                "horizon": "emergence",
                "resonance": 0.5,
                "velocity": 0.5,
                "density": 0.5,
            },
        }

        detector = EntanglementDetector()
        score = detector.score(bc1, bc2)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_validation_result_passed_property(self):
        """ValidationResult.passed should check thresholds."""
        from fractalsemantics.exp06_entanglement_detection import ValidationResult

        # Passing result
        result_pass = ValidationResult(
            threshold=0.85,
            true_positives=90,
            false_positives=5,
            false_negatives=10,
            true_negatives=95,
            precision=0.95,
            recall=0.90,
            f1_score=0.92,
            accuracy=0.92,
            runtime_seconds=1.0,
        )

        assert result_pass.passed

        # Failing result
        result_fail = ValidationResult(
            threshold=0.85,
            true_positives=50,
            false_positives=50,
            false_negatives=50,
            true_negatives=50,
            precision=0.50,
            recall=0.50,
            f1_score=0.50,
            accuracy=0.50,
            runtime_seconds=1.0,
        )

        assert not result_fail.passed

    def test_entanglement_detector_threshold_validation(self):
        """EntanglementDetector should validate threshold."""
        from fractalsemantics.exp06_entanglement_detection import (
            EntanglementDetector,
        )

        with pytest.raises(ValueError, match="Threshold must be in"):
            EntanglementDetector(threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be in"):
            EntanglementDetector(threshold=-0.1)

    def test_compute_validation_metrics_empty_sets(self):
        """compute_validation_metrics should handle empty sets."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_validation_metrics,
        )

        result = compute_validation_metrics(set(), set(), 100)

        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.true_negatives == 100
