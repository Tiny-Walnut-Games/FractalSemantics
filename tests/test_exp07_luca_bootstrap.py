"""
Test suite for EXP-07: LUCA Bootstrap Test
Tests system reconstruction from minimal LUCA representation.
"""

import json


class TestTestBitChain:
    """Test TestBitChain entity for LUCA testing."""

    def test_test_bitchain_initialization(self):
        """TestBitChain should initialize with default values."""
        from fractalsemantics.exp07_luca_bootstrap import TestBitChain

        bc = TestBitChain()
        assert bc is not None
        assert bc.content == ""
        assert bc.lineage == 0
        assert bc.realm == "pattern"

    def test_test_bitchain_custom_values(self):
        """TestBitChain should accept custom initialization values."""
        from fractalsemantics.exp07_luca_bootstrap import TestBitChain

        bc = TestBitChain(
            content="Test content",
            lineage=5,
            realm="data",
            horizon="peak",
        )

        assert bc.content == "Test content"
        assert bc.lineage == 5
        assert bc.realm == "data"
        assert bc.horizon == "peak"

    def test_test_bitchain_to_dict(self):
        """TestBitChain should convert to dictionary."""
        from fractalsemantics.exp07_luca_bootstrap import TestBitChain

        bc = TestBitChain(content="test", lineage=2)
        bc_dict = bc.to_dict()

        assert isinstance(bc_dict, dict)
        assert "bit_chain_id" in bc_dict
        assert "content" in bc_dict
        assert bc_dict["lineage"] == 2

    def test_test_bitchain_to_json(self):
        """TestBitChain should convert to JSON string."""
        from fractalsemantics.exp07_luca_bootstrap import TestBitChain

        bc = TestBitChain(content="json test")
        json_str = bc.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["content"] == "json test"

    def test_test_bitchain_fractalsemantics_address(self):
        """TestBitChain should generate FractalSemantics address."""
        from fractalsemantics.exp07_luca_bootstrap import TestBitChain

        bc = TestBitChain(lineage=3, realm="data", horizon="emergence")
        address = bc.get_fractalsemantics_address()

        assert isinstance(address, str)
        assert address.startswith("FractalSemantics-")
        assert "003" in address


class TestLUCABootstrapResult:
    """Test LUCA bootstrap result tracking."""

    def test_result_initialization(self):
        """LUCABootstrapResult should initialize with defaults."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapResult

        result = LUCABootstrapResult()
        assert result.experiment == "EXP-07"
        assert result.status == "PASS"
        assert isinstance(result.results, dict)

    def test_result_to_dict(self):
        """LUCABootstrapResult should serialize to dict."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapResult

        result = LUCABootstrapResult(
            status="PASS",
            results={"test_key": "test_value"},
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["experiment"] == "EXP-07"
        assert "results" in result_dict


class TestLUCABootstrapTester:
    """Test LUCA bootstrap tester."""

    def test_tester_initialization(self):
        """LUCABootstrapTester should initialize."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        assert tester is not None
        assert isinstance(tester.luca_dictionary, dict)

    def test_create_test_entities(self):
        """create_test_entities should generate entities with lineage."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=5)

        assert len(entities) == 5
        assert all(e.lineage > 0 for e in entities)

    def test_entities_have_varied_properties(self):
        """Created entities should have varied realms and horizons."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=10)

        realms = set(e.realm for e in entities)
        horizons = set(e.horizon for e in entities)

        assert len(realms) > 1, "Should have varied realms"
        assert len(horizons) > 1, "Should have varied horizons"

    def test_compute_luca_encoding(self):
        """compute_luca_encoding should create minimal form."""
        from fractalsemantics.exp07_luca_bootstrap import (
            LUCABootstrapTester,
            TestBitChain,
        )

        tester = LUCABootstrapTester()
        entity = TestBitChain(
            content="Test entity",
            lineage=3,
            realm="data",
        )

        luca_form = tester.compute_luca_encoding(entity)

        assert isinstance(luca_form, dict)
        assert "id" in luca_form
        assert "hash" in luca_form
        assert "lineage" in luca_form

    def test_compress_to_luca(self):
        """compress_to_luca should create LUCA state."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=5)

        luca_state = tester.compress_to_luca(entities)

        assert isinstance(luca_state, dict)
        assert "entity_count" in luca_state
        assert luca_state["entity_count"] == 5
        assert "encodings" in luca_state
        assert len(luca_state["encodings"]) == 5

    def test_luca_state_compression_ratio(self):
        """LUCA state should track compression ratio."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=3)

        luca_state = tester.compress_to_luca(entities)

        assert "compression_ratio" in luca_state
        assert luca_state["compression_ratio"] > 0

    def test_bootstrap_from_luca(self):
        """bootstrap_from_luca should reconstruct entities."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=5)
        luca_state = tester.compress_to_luca(original)

        bootstrapped, success_flags = tester.bootstrap_from_luca(luca_state)

        assert len(bootstrapped) == 5
        assert len(success_flags) == 5
        assert all(success_flags)

    def test_compare_entities(self):
        """compare_entities should match original and bootstrapped."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=5)
        luca_state = tester.compress_to_luca(original)
        bootstrapped, _ = tester.bootstrap_from_luca(luca_state)

        comparison = tester.compare_entities(original, bootstrapped)

        assert isinstance(comparison, dict)
        assert "count_match" in comparison
        assert comparison["count_match"]

    def test_comparison_tracks_matches(self):
        """Comparison should count matching attributes."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=3)
        luca_state = tester.compress_to_luca(original)
        bootstrapped, _ = tester.bootstrap_from_luca(luca_state)

        comparison = tester.compare_entities(original, bootstrapped)

        assert "id_matches" in comparison
        assert "lineage_matches" in comparison
        assert comparison["id_matches"] >= 0

    def test_test_fractal_properties(self):
        """test_fractal_properties should verify self-similarity."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        entities = tester.create_test_entities(num_entities=5)

        fractal_results = tester.test_fractal_properties(entities)

        assert isinstance(fractal_results, dict)
        assert "self_similarity" in fractal_results
        assert "scale_invariance" in fractal_results

    def test_test_luca_continuity(self):
        """test_luca_continuity should verify lineage integrity."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=5)

        continuity = tester.test_luca_continuity(original)

        assert isinstance(continuity, dict)
        assert "lineage_continuity" in continuity


class TestLUCABootstrapIntegration:
    """Test complete LUCA bootstrap workflow."""

    def test_full_bootstrap_cycle(self):
        """Full cycle: create → compress → bootstrap → compare."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()

        original = tester.create_test_entities(num_entities=5)
        luca_state = tester.compress_to_luca(original)
        bootstrapped, success = tester.bootstrap_from_luca(luca_state)
        comparison = tester.compare_entities(original, bootstrapped)

        assert len(bootstrapped) == len(original)
        assert all(success)
        assert comparison["entity_recovery_rate"] == 1.0

    def test_bootstrap_preserves_lineage(self):
        """Bootstrap should preserve lineage through compression."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=10)

        lineages_original = [e.lineage for e in original]
        luca_state = tester.compress_to_luca(original)
        bootstrapped, _ = tester.bootstrap_from_luca(luca_state)
        lineages_bootstrapped = [e.lineage for e in bootstrapped]

        assert lineages_original == lineages_bootstrapped

    def test_luca_acts_as_recovery_point(self):
        """LUCA should enable system recovery after compression."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()

        original_entities = tester.create_test_entities(num_entities=7)
        compression = tester.compress_to_luca(original_entities)

        assert len(compression["encodings"]) == 7

        reconstructed, all_success = tester.bootstrap_from_luca(compression)

        assert all(all_success)
        assert len(reconstructed) == len(original_entities)


class TestErrorHandling:
    """Test error handling and robustness."""

    def test_bootstrap_with_corrupt_encoding(self):
        """Test bootstrap resilience with corrupted LUCA encoding."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester, TestBitChain

        tester = LUCABootstrapTester()
        entity = TestBitChain(content="test", lineage=1)
        original_encoding = tester.compute_luca_encoding(entity)

        # Corrupt the encoding
        corrupt_encoding = original_encoding.copy()
        corrupt_encoding["realm_sig"] = "X"  # Invalid signature

        luca_state = {"encodings": [corrupt_encoding]}

        # Bootstrap should handle corruption gracefully
        bootstrapped, success_flags = tester.bootstrap_from_luca(luca_state)

        assert len(bootstrapped) == 1
        assert len(success_flags) == 1
        # Should still succeed but with "unknown" realm
        assert bootstrapped[0].realm == "unknown"

    def test_empty_entity_list(self):
        """Test behavior with empty entity lists."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()

        # Empty compression
        luca_state = tester.compress_to_luca([])
        assert luca_state["entity_count"] == 0
        assert len(luca_state["encodings"]) == 0

        # Empty bootstrap
        bootstrapped, success = tester.bootstrap_from_luca(luca_state)
        assert len(bootstrapped) == 0
        assert len(success) == 0

    def test_large_scale_bootstrap(self):
        """Test bootstrap with larger entity sets."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()

        # Test with larger set
        original = tester.create_test_entities(num_entities=50)
        luca_state = tester.compress_to_luca(original)
        bootstrapped, success = tester.bootstrap_from_luca(luca_state)

        assert len(bootstrapped) == 50
        assert all(success)

        # Verify compression is effective
        assert luca_state["compression_ratio"] > 0
        assert luca_state["compression_ratio"] < 1.0

    def test_lineage_preservation_under_stress(self):
        """Test lineage integrity under repeated bootstrap cycles."""
        from fractalsemantics.exp07_luca_bootstrap import LUCABootstrapTester

        tester = LUCABootstrapTester()
        original = tester.create_test_entities(num_entities=20)

        # Multiple bootstrap cycles
        current = original
        for cycle in range(5):
            luca_state = tester.compress_to_luca(current)
            current, success = tester.bootstrap_from_luca(luca_state)
            assert all(success), f"Failed at cycle {cycle}"

        # Verify final lineage matches original
        original_lineages = sorted([e.lineage for e in original])
        final_lineages = sorted([e.lineage for e in current])
        assert original_lineages == final_lineages
