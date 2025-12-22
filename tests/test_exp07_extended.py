"""
Extended tests for EXP-07 LUCA Bootstrap to achieve 95%+ coverage
"""
# pylint: disable=protected-access

import tempfile
import json
from pathlib import Path


class TestExp07Extended:
    """Extended tests for LUCA bootstrap experiment."""

    def test_run_comprehensive_test_execution(self):
        """run_comprehensive_test should execute full test."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        results = tester.run_comprehensive_test()

        assert results.experiment == "EXP-07"
        assert results.status in ["PASS", "FAIL"]
        assert "compression" in results.results

    def test_save_results_file_io(self):
        """Results should be saveable to file."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapResult

        result = LUCABootstrapResult(status="PASS", results={"test": "data"})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_results.json"

            with open(output_file, "w") as f:
                json.dump(result.to_dict(), f)

            assert output_file.exists()

            with open(output_file) as f:
                data = json.load(f)
                assert data["experiment"] == "EXP-07"

    def test_error_handling_in_bootstrap_cycles(self):
        """Bootstrap should handle errors gracefully."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()

        # Create entities with potential edge cases
        entities = tester.create_test_entities(num_entities=3)

        # Test continuity with small dataset
        continuity = tester.test_luca_continuity(entities)

        assert "bootstraps_performed" in continuity
        assert "bootstrap_failures" in continuity

    def test_main_entry_point(self):
        """Main entry point should execute successfully."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        results = tester.run_comprehensive_test()

        assert results is not None
        assert isinstance(results.results, dict)

    def test_expand_signature_all_values(self):
        """_expand_signature should handle all signature values."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()

        assert tester._expand_signature("p") == "pattern"
        assert tester._expand_signature("d") == "data"
        assert tester._expand_signature("n") == "narrative"
        assert tester._expand_signature("e") == "emergence"
        assert tester._expand_signature("k") == "peak"
        assert tester._expand_signature("c") == "crystallization"
        assert tester._expand_signature("l") == "logic"
        assert tester._expand_signature("r") == "creativity"
        assert tester._expand_signature("x") == "unknown"

    def test_luca_encoding_completeness(self):
        """LUCA encoding should preserve essential information."""
        from fractalstat.exp07_luca_bootstrap import (
            LUCABootstrapTester,
            TestBitChain,
        )

        tester = LUCABootstrapTester()
        entity = TestBitChain(
            content="Test content",
            lineage=5,
            realm="data",
            horizon="peak",
            polarity="logic",
            dimensionality=3,
        )

        luca_form = tester.compute_luca_encoding(entity)

        assert "id" in luca_form
        assert "hash" in luca_form
        assert "lineage" in luca_form
        assert luca_form["lineage"] == 5
        assert luca_form["realm_sig"] == "d"
        assert luca_form["horizon_sig"] == "p"

    def test_bootstrap_reconstruction_accuracy(self):
        """Bootstrap should accurately reconstruct entities."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=5)

        luca_state = tester.compress_to_luca(original)
        bootstrapped, success = tester.bootstrap_from_luca(luca_state)

        # All should succeed
        assert all(success)

        # Lineages should match
        for orig, boot in zip(original, bootstrapped):
            assert orig.lineage == boot.lineage

    def test_fractal_properties_validation(self):
        """Fractal properties should be validated correctly."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=10)

        fractal_tests = tester.test_fractal_properties(entities)

        assert "self_similarity" in fractal_tests
        assert "scale_invariance" in fractal_tests
        assert "recursive_structure" in fractal_tests
        assert "luca_traceability" in fractal_tests
        assert "details" in fractal_tests

    def test_comparison_with_mismatched_counts(self):
        """Comparison should handle mismatched entity counts."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=5)
        bootstrapped = tester.create_test_entities(num_entities=3)

        comparison = tester.compare_entities(original, bootstrapped)

        assert not comparison["count_match"]
        assert comparison["original_count"] == 5
        assert comparison["bootstrapped_count"] == 3

    def test_luca_state_hash_consistency(self):
        """LUCA state hash should be consistent for same input."""
        from fractalstat.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=5)

        luca_state1 = tester.compress_to_luca(entities)
        luca_state2 = tester.compress_to_luca(entities)

        # Hashes should be identical for same entities
        assert luca_state1["luca_hash"] == luca_state2["luca_hash"]
