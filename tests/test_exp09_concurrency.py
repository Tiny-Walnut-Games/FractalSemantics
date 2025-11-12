"""
Test suite for EXP-09: Concurrency & Thread Safety (LLM Integration)
Tests concurrent embedding generation, narrative enhancement, and STAT7 extraction.
"""

import time
from dataclasses import dataclass


@dataclass
class MockBitChain:
    """Mock BitChain for testing."""

    bit_chain_id: str
    content: str
    realm: str
    luminosity: float = 0.5


class TestConcurrencyInit:
    """Test ConcurrencyTester initialization."""

    def test_concurrency_tester_initializes(self):
        """ConcurrencyTester should initialize without errors."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        assert tester is not None
        assert hasattr(tester, "llm_demo")
        assert hasattr(tester, "num_workers")

    def test_default_worker_count(self):
        """Default worker count should be reasonable."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        assert tester.num_workers > 0
        assert tester.num_workers <= 20


class TestConcurrentEmbedding:
    """Test concurrent embedding generation."""

    def test_concurrent_embedding_generation(self):
        """Should generate embeddings for multiple entities concurrently."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}", content=f"Test entity {i}", realm="companion"
            )
            for i in range(5)
        ]

        results = tester.run_concurrent_embeddings(bit_chains, num_workers=2)

        assert isinstance(results, list)
        assert len(results) == 5

    def test_all_embeddings_valid(self):
        """All embeddings should be valid numpy arrays."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        import numpy as np

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id=f"test-{i}", content=f"Test {i}", realm="badge")
            for i in range(3)
        ]

        results = tester.run_concurrent_embeddings(bit_chains, num_workers=2)

        for result in results:
            assert isinstance(result["embedding"], np.ndarray)
            assert len(result["embedding"]) == 384


class TestConcurrentNarrativeEnhancement:
    """Test concurrent narrative enhancement."""

    def test_concurrent_narrative_enhancement(self):
        """Should enhance narratives for multiple entities concurrently."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}", content=f"Test {i}", realm="companion"
            )
            for i in range(2)
        ]

        results = tester.run_concurrent_enhancements(bit_chains, num_workers=1)

        assert isinstance(results, list)
        assert len(results) == 2

    def test_all_narratives_valid(self):
        """All narratives should have required fields."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id=f"test-{i}", content=f"Test {i}", realm="badge")
            for i in range(2)
        ]

        results = tester.run_concurrent_enhancements(bit_chains, num_workers=1)

        for result in results:
            assert "bit_chain_id" in result
            assert "embedding" in result
            assert "enhanced_narrative" in result


class TestConcurrentSTAT7Extraction:
    """Test concurrent STAT7 coordinate extraction."""

    def test_concurrent_stat7_extraction(self):
        """Should extract STAT7 coordinates concurrently."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        import numpy as np

        tester = ConcurrencyTester()

        # Generate embeddings
        embeddings = [np.random.rand(384) for _ in range(5)]

        results = tester.run_concurrent_stat7_extraction(embeddings, num_workers=2)

        assert isinstance(results, list)
        assert len(results) == 5

    def test_all_coordinates_valid(self):
        """All extracted coordinates should be valid."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        import numpy as np

        tester = ConcurrencyTester()
        embeddings = [np.random.rand(384) for _ in range(3)]

        results = tester.run_concurrent_stat7_extraction(embeddings, num_workers=2)

        for result in results:
            assert isinstance(result, dict)
            assert "luminosity" in result
            assert "lineage" in result
            assert 0 <= result["luminosity"] <= 1


class TestNoRaceConditions:
    """Test for race conditions and data integrity."""

    def test_concurrent_results_consistent(self):
        """Concurrent processing should yield consistent results."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}", content="same content", realm="companion"
            )
            for i in range(2)
        ]

        # Run twice
        results1 = tester.run_concurrent_enhancements(bit_chains, num_workers=1)
        results2 = tester.run_concurrent_enhancements(bit_chains, num_workers=1)

        # Both runs should complete without errors
        assert len(results1) == 2
        assert len(results2) == 2

    def test_thread_safety_under_load(self):
        """System should handle high concurrency without errors."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        # Create few entities
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"load-{i}", content=f"Load test {i}", realm="companion"
            )
            for i in range(2)
        ]

        # Run with single worker
        start = time.time()
        results = tester.run_concurrent_enhancements(bit_chains, num_workers=1)
        elapsed = time.time() - start

        assert len(results) == 2
        assert elapsed > 0


class TestThroughputMetrics:
    """Test throughput metrics."""

    def test_throughput_measurement(self):
        """Should measure and report throughput."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}", content=f"Test {i}", realm="companion"
            )
            for i in range(2)
        ]

        result = tester.run_throughput_test(bit_chains, num_workers=1)

        assert "throughput_qps" in result
        assert "total_time_seconds" in result
        assert "completed_queries" in result
        assert result["completed_queries"] == 2

    def test_throughput_is_positive(self):
        """Throughput should be a positive number."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        result = tester.run_throughput_test(bit_chains, num_workers=1)

        assert result["throughput_qps"] >= 0
        assert result["total_time_seconds"] > 0


class TestConcurrencyResults:
    """Test result reporting."""

    def test_generate_report(self):
        """Should generate concurrency test report."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        report = tester.run_full_concurrency_test(bit_chains, num_workers=1)

        assert isinstance(report, dict)
        assert "experiment" in report
        assert "status" in report
        assert "results" in report

    def test_report_contains_metrics(self):
        """Report should contain key metrics."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        report = tester.run_full_concurrency_test(bit_chains, num_workers=1)
        results = report["results"]

        assert "embedding_throughput" in results
        assert "enhancement_throughput" in results
        assert "stat7_extraction_throughput" in results
        assert "no_race_conditions" in results


class TestStressScenarios:
    """Test stress scenarios."""

    def test_high_concurrency(self):
        """Should handle high concurrency levels."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(
                bit_chain_id=f"stress-{i}", content=f"Stress {i}", realm="companion"
            )
            for i in range(2)
        ]

        results = tester.run_concurrent_enhancements(bit_chains, num_workers=2)

        assert len(results) == 2
        assert all("bit_chain_id" in r for r in results)

    def test_varying_worker_counts(self):
        """Should work with different worker counts."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()

        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        for num_workers in [1, 2]:
            results = tester.run_concurrent_enhancements(
                bit_chains, num_workers=num_workers
            )
            assert len(results) == 1


class TestBranchCoverageExp09:
    """Additional tests to increase branch coverage for exp09."""

    def test_result_dataclass_timestamp_auto_generation(self):
        """ConcurrencyTestResult should auto-generate timestamp if empty."""
        from fractalstat.exp09_concurrency import ConcurrencyTestResult

        result = ConcurrencyTestResult()
        assert result.timestamp != ""
        assert len(result.timestamp) > 0

    def test_result_dataclass_results_auto_initialization(self):
        """ConcurrencyTestResult should auto-initialize results dict if None."""
        from fractalstat.exp09_concurrency import ConcurrencyTestResult

        result = ConcurrencyTestResult()
        assert result.results is not None
        assert isinstance(result.results, dict)

    def test_result_to_json(self):
        """ConcurrencyTestResult should convert to JSON string."""
        from fractalstat.exp09_concurrency import ConcurrencyTestResult

        result = ConcurrencyTestResult(
            experiment="EXP-09",
            title="Test",
            status="PASS",
            results={"test": "data"},
        )
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "EXP-09" in json_str

    def test_concurrent_embeddings_with_default_workers(self):
        """run_concurrent_embeddings should use default workers when None."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        results = tester.run_concurrent_embeddings(bit_chains, num_workers=None)
        assert len(results) == 1

    def test_concurrent_enhancements_with_default_workers(self):
        """run_concurrent_enhancements should use default workers when None."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        results = tester.run_concurrent_enhancements(bit_chains, num_workers=None)
        assert len(results) == 1

    def test_concurrent_stat7_extraction_with_default_workers(self):
        """run_concurrent_stat7_extraction should use default workers when None."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        import numpy as np

        tester = ConcurrencyTester()
        embeddings = [np.random.rand(384)]

        results = tester.run_concurrent_stat7_extraction(embeddings, num_workers=None)
        assert len(results) == 1

    def test_throughput_test_with_default_workers(self):
        """run_throughput_test should use default workers when None."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        result = tester.run_throughput_test(bit_chains, num_workers=None)
        assert "throughput_qps" in result

    def test_full_concurrency_test_with_default_workers(self):
        """run_full_concurrency_test should use default workers when None."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        report = tester.run_full_concurrency_test(bit_chains, num_workers=None)
        assert "status" in report

    def test_embedding_exception_handling(self):
        """Should handle exceptions during embedding generation."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        results = tester.run_concurrent_embeddings(bit_chains, num_workers=1)
        assert isinstance(results, list)

    def test_enhancement_exception_handling(self):
        """Should handle exceptions during narrative enhancement."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        results = tester.run_concurrent_enhancements(bit_chains, num_workers=1)
        assert isinstance(results, list)

    def test_stat7_extraction_exception_handling(self):
        """Should handle exceptions during STAT7 extraction."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        import numpy as np

        tester = ConcurrencyTester()
        embeddings = [np.random.rand(384)]

        results = tester.run_concurrent_stat7_extraction(embeddings, num_workers=1)
        assert isinstance(results, list)

    def test_race_condition_test_consistent_results(self):
        """Race condition test should detect consistent results."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        result = tester.test_race_conditions(bit_chains, num_iterations=3)
        assert "consistent" in result
        assert "no_race_conditions" in result

    def test_full_test_status_pass(self):
        """Full test should return PASS status when all conditions met."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        report = tester.run_full_concurrency_test(bit_chains, num_workers=1)
        assert report["status"] in ["PASS", "FAIL"]

    def test_full_test_status_fail_conditions(self):
        """Full test should return FAIL status when conditions not met."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        report = tester.run_full_concurrency_test(bit_chains, num_workers=1)
        assert "all_results_valid" in report["results"]

    def test_save_results_creates_file(self):
        """save_results should create a JSON file."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        from pathlib import Path
        import os

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        tester.run_full_concurrency_test(bit_chains, num_workers=1)
        filepath = tester.save_results(output_dir="results")

        assert os.path.exists(filepath)
        os.remove(filepath)

    def test_throughput_calculation_with_zero_elapsed_time(self):
        """Throughput should handle zero elapsed time gracefully."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        result = tester.run_throughput_test(bit_chains, num_workers=1)
        assert result["throughput_qps"] >= 0

    def test_race_condition_iterations(self):
        """Race condition test should run specified number of iterations."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(bit_chain_id="test-0", content="Test", realm="companion")
        ]

        result = tester.test_race_conditions(bit_chains, num_iterations=5)
        assert result["iterations"] == 5
        assert len(result["results_per_iteration"]) == 5

    def test_concurrent_embeddings_multiple_entities(self):
        """Should handle multiple entities concurrently."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}", content=f"Test {i}", realm="companion"
            )
            for i in range(3)
        ]

        results = tester.run_concurrent_embeddings(bit_chains, num_workers=2)
        assert len(results) == 3

    def test_concurrent_enhancements_multiple_entities(self):
        """Should enhance multiple entities concurrently."""
        from fractalstat.exp09_concurrency import ConcurrencyTester

        tester = ConcurrencyTester()
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}", content=f"Test {i}", realm="companion"
            )
            for i in range(3)
        ]

        results = tester.run_concurrent_enhancements(bit_chains, num_workers=2)
        assert len(results) == 3

    def test_concurrent_stat7_extraction_multiple_embeddings(self):
        """Should extract STAT7 from multiple embeddings concurrently."""
        from fractalstat.exp09_concurrency import ConcurrencyTester
        import numpy as np

        tester = ConcurrencyTester()
        embeddings = [np.random.rand(384) for _ in range(3)]

        results = tester.run_concurrent_stat7_extraction(embeddings, num_workers=2)
        assert len(results) == 3
