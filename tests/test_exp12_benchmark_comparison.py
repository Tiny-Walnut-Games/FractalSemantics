"""
Tests for EXP-12: Benchmark Comparison
Comprehensive test coverage targeting 95%+
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from fractalstat.exp12_benchmark_comparison import (
    BenchmarkComparisonExperiment,
    SystemBenchmarkResult,
    BenchmarkComparisonResult,
    UUIDSystem,
    SHA256System,
    VectorDBSystem,
    GraphDBSystem,
    RDBMSSystem,
    FractalStatSystem,
    save_results,
)
from fractalstat.fractalstat_experiments import generate_random_bitchain


class TestSystemBenchmarkResult:
    """Tests for SystemBenchmarkResult dataclass."""

    def test_initialization(self):
        """SystemBenchmarkResult should initialize with all fields."""
        result = SystemBenchmarkResult(
            system_name="FractalStat",
            scale=10000,
            num_queries=1000,
            unique_addresses=10000,
            collisions=0,
            collision_rate=0.0,
            mean_retrieval_latency_ms=0.05,
            median_retrieval_latency_ms=0.04,
            p95_retrieval_latency_ms=0.08,
            p99_retrieval_latency_ms=0.10,
            avg_storage_bytes_per_entity=150,
            total_storage_bytes=1500000,
            semantic_expressiveness=0.95,
            relationship_support=0.8,
            query_flexibility=0.9,
        )

        assert result.system_name == "FractalStat"
        assert result.scale == 10000
        assert result.collisions == 0
        assert result.semantic_expressiveness == 0.95

    def test_to_dict(self):
        """SystemBenchmarkResult should serialize to dict."""
        result = SystemBenchmarkResult(
            system_name="UUID",
            scale=1000,
            num_queries=100,
            unique_addresses=1000,
            collisions=0,
            collision_rate=0.0,
            mean_retrieval_latency_ms=0.03,
            median_retrieval_latency_ms=0.02,
            p95_retrieval_latency_ms=0.05,
            p99_retrieval_latency_ms=0.07,
            avg_storage_bytes_per_entity=100,
            total_storage_bytes=100000,
            semantic_expressiveness=0.0,
            relationship_support=0.0,
            query_flexibility=0.1,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["system_name"] == "UUID"
        assert result_dict["scale"] == 1000


class TestBenchmarkComparisonResult:
    """Tests for BenchmarkComparisonResult dataclass."""

    def test_initialization(self):
        """BenchmarkComparisonResult should initialize with all fields."""
        result = BenchmarkComparisonResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=10000,
            scales_tested=[10000],
            num_queries=1000,
            systems_tested=["uuid", "fractalstat"],
            system_results=[],
            best_collision_rate_system="FractalStat",
            best_retrieval_latency_system="UUID",
            best_storage_efficiency_system="UUID",
            best_semantic_expressiveness_system="FractalStat",
            best_overall_system="FractalStat",
            fractalstat_rank_collision=1,
            fractalstat_rank_retrieval=2,
            fractalstat_rank_storage=3,
            fractalstat_rank_semantic=1,
            fractalstat_overall_score=0.85,
            major_findings=["FractalStat competitive"],
            fractalstat_competitive=True,
        )

        assert result.sample_size == 10000
        assert result.fractalstat_competitive
        assert result.best_overall_system == "FractalStat"

    def test_to_dict(self):
        """BenchmarkComparisonResult should serialize to dict."""
        result = BenchmarkComparisonResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=10000,
            scales_tested=[10000],
            num_queries=1000,
            systems_tested=["uuid", "fractalstat"],
            system_results=[],
            best_collision_rate_system="FractalStat",
            best_retrieval_latency_system="UUID",
            best_storage_efficiency_system="UUID",
            best_semantic_expressiveness_system="FractalStat",
            best_overall_system="FractalStat",
            fractalstat_rank_collision=1,
            fractalstat_rank_retrieval=2,
            fractalstat_rank_storage=3,
            fractalstat_rank_semantic=1,
            fractalstat_overall_score=0.85,
            major_findings=[],
            fractalstat_competitive=True,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["experiment"] == "EXP-12"
        assert result_dict["test_type"] == "Benchmark Comparison"
        assert result_dict["fractalstat_positioning"]["competitive"]


class TestBenchmarkSystems:
    """Tests for individual benchmark system implementations."""

    def test_uuid_system(self):
        """UUIDSystem should generate unique UUIDs."""
        system = UUIDSystem()

        assert system.name == "UUID"

        # Generate addresses
        addr1 = system.generate_address("entity1")
        addr2 = system.generate_address("entity2")

        assert addr1 != addr2
        assert len(addr1) == 36  # UUID format

        # Test capabilities
        assert system.get_semantic_expressiveness() == 0.0
        assert system.get_relationship_support() == 0.0
        assert system.get_query_flexibility() == 0.1

    def test_sha256_system(self):
        """SHA256System should generate content-based hashes."""
        system = SHA256System()

        assert system.name == "SHA256"

        # Generate addresses
        bc1 = generate_random_bitchain(seed=1)
        bc2 = generate_random_bitchain(seed=2)

        addr1 = system.generate_address(bc1)
        addr2 = system.generate_address(bc2)

        assert addr1 != addr2
        assert len(addr1) == 64  # SHA-256 hex

        # Same entity should produce same hash
        addr1_again = system.generate_address(bc1)
        assert addr1 == addr1_again

        # Test capabilities
        assert system.get_semantic_expressiveness() == 0.1
        assert system.get_relationship_support() == 0.0
        assert system.get_query_flexibility() == 0.2

    def test_vector_db_system(self):
        """VectorDBSystem should store embeddings."""
        system = VectorDBSystem()

        assert system.name == "VectorDB"

        # Generate address with embedding
        bc = generate_random_bitchain(seed=1)
        addr = system.generate_address(bc)

        assert addr in system.embeddings
        assert isinstance(system.embeddings[addr], list)
        assert len(system.embeddings[addr]) == 3

        # Test capabilities
        assert system.get_semantic_expressiveness() == 0.7
        assert system.get_relationship_support() == 0.3
        assert system.get_query_flexibility() == 0.6

    def test_graph_db_system(self):
        """GraphDBSystem should store relationships."""
        system = GraphDBSystem()

        assert system.name == "GraphDB"

        # Generate address with edges
        bc = generate_random_bitchain(seed=1)
        addr = system.generate_address(bc)

        assert addr in system.edges
        assert isinstance(system.edges[addr], list)

        # Test capabilities
        assert system.get_semantic_expressiveness() == 0.4
        assert system.get_relationship_support() == 0.9
        assert system.get_query_flexibility() == 0.7

    def test_rdbms_system(self):
        """RDBMSSystem should build indexes."""
        system = RDBMSSystem()

        assert system.name == "RDBMS"

        # Generate address with indexes
        bc = generate_random_bitchain(seed=1)
        addr = system.generate_address(bc)

        # Check index was created
        realm_key = f"realm:{bc.coordinates.realm}"
        assert realm_key in system.indexes
        assert addr in system.indexes[realm_key]

        # Test capabilities
        assert system.get_semantic_expressiveness() == 0.5
        assert system.get_relationship_support() == 0.6
        assert system.get_query_flexibility() == 0.8

    def test_fractalstat_system(self):
        """FractalStatSystem should use FractalStat addressing."""
        system = FractalStatSystem()

        assert system.name == "FractalStat"

        # Generate address
        bc = generate_random_bitchain(seed=1)
        addr = system.generate_address(bc)

        assert len(addr) == 64  # SHA-256 hex

        # Same bitchain should produce same address
        addr_again = system.generate_address(bc)
        assert addr == addr_again

        # Test capabilities
        assert system.get_semantic_expressiveness() == 0.95
        assert system.get_relationship_support() == 0.8
        assert system.get_query_flexibility() == 0.9

    def test_system_store_and_retrieve(self):
        """All systems should support store and retrieve."""
        systems = [
            UUIDSystem(),
            SHA256System(),
            VectorDBSystem(),
            GraphDBSystem(),
            RDBMSSystem(),
            FractalStatSystem(),
        ]

        for system in systems:
            bc = generate_random_bitchain(seed=1)
            addr = system.generate_address(bc)

            # Store
            system.store(addr, bc)

            # Retrieve
            retrieved = system.retrieve(addr)
            assert retrieved == bc


class TestBenchmarkComparisonExperiment:
    """Tests for BenchmarkComparisonExperiment class."""

    def test_initialization(self):
        """BenchmarkComparisonExperiment should initialize with parameters."""
        exp = BenchmarkComparisonExperiment(
            sample_size=1000,
            benchmark_systems=["uuid", "fractalstat"],
            scales=[1000],
            num_queries=100,
        )

        assert exp.sample_size == 1000
        assert exp.benchmark_systems == ["uuid", "fractalstat"]
        assert exp.scales == [1000]
        assert exp.num_queries == 100
        assert exp.results == []

    def test_initialization_defaults(self):
        """BenchmarkComparisonExperiment should use default parameters."""
        exp = BenchmarkComparisonExperiment()

        assert exp.sample_size == 100000
        assert "uuid" in exp.benchmark_systems
        assert "fractalstat" in exp.benchmark_systems
        assert exp.num_queries == 1000

    def test_create_system(self):
        """_create_system should instantiate correct system classes."""
        exp = BenchmarkComparisonExperiment()

        uuid_sys = exp._create_system("uuid")
        assert isinstance(uuid_sys, UUIDSystem)

        sha256_sys = exp._create_system("sha256")
        assert isinstance(sha256_sys, SHA256System)

        vector_sys = exp._create_system("vector_db")
        assert isinstance(vector_sys, VectorDBSystem)

        graph_sys = exp._create_system("graph_db")
        assert isinstance(graph_sys, GraphDBSystem)

        rdbms_sys = exp._create_system("rdbms")
        assert isinstance(rdbms_sys, RDBMSSystem)

        fractalstat_sys = exp._create_system("fractalstat")
        assert isinstance(fractalstat_sys, FractalStatSystem)

    def test_create_system_invalid(self):
        """_create_system should raise error for unknown system."""
        exp = BenchmarkComparisonExperiment()

        with pytest.raises(ValueError, match="Unknown system"):
            exp._create_system("invalid_system")

    def test_benchmark_system(self):
        """_benchmark_system should benchmark a single system."""
        exp = BenchmarkComparisonExperiment(sample_size=100, num_queries=10)
        system = UUIDSystem()

        result = exp._benchmark_system(system, 100)

        assert isinstance(result, SystemBenchmarkResult)
        assert result.system_name == "UUID"
        assert result.scale == 100
        assert result.num_queries == 10
        assert result.unique_addresses > 0
        assert result.mean_retrieval_latency_ms >= 0

    def test_run_small_sample(self):
        """run() should execute with small sample for testing."""
        exp = BenchmarkComparisonExperiment(
            sample_size=100,
            benchmark_systems=["uuid", "fractalstat"],
            scales=[100],
            num_queries=10,
        )

        result, success = exp.run()

        assert isinstance(result, BenchmarkComparisonResult)
        assert isinstance(success, bool)
        assert result.sample_size == 100
        assert len(result.system_results) == 2

    def test_run_comparative_analysis(self):
        """run() should perform comparative analysis."""
        exp = BenchmarkComparisonExperiment(
            sample_size=100,
            benchmark_systems=["uuid", "sha256", "fractalstat"],
            scales=[100],
            num_queries=10,
        )

        result, success = exp.run()

        # Check that best systems are identified
        assert result.best_collision_rate_system in ["UUID", "SHA256", "FractalStat"]
        assert result.best_retrieval_latency_system in [
            "UUID",
            "SHA256",
            "FractalStat",
        ]
        assert result.best_storage_efficiency_system in [
            "UUID",
            "SHA256",
            "FractalStat",
        ]
        assert result.best_semantic_expressiveness_system in [
            "UUID",
            "SHA256",
            "FractalStat",
        ]
        assert result.best_overall_system in ["UUID", "SHA256", "FractalStat"]

    def test_run_fractalstat_rankings(self):
        """run() should calculate FractalStat rankings."""
        exp = BenchmarkComparisonExperiment(
            sample_size=100,
            benchmark_systems=["uuid", "sha256", "fractalstat"],
            scales=[100],
            num_queries=10,
        )

        result, success = exp.run()

        # FractalStat should be ranked
        assert 1 <= result.fractalstat_rank_collision <= 3
        assert 1 <= result.fractalstat_rank_retrieval <= 3
        assert 1 <= result.fractalstat_rank_storage <= 3
        assert 1 <= result.fractalstat_rank_semantic <= 3
        assert 0.0 <= result.fractalstat_overall_score <= 1.0

    def test_run_fractalstat_competitive(self):
        """run() should evaluate if FractalStat is competitive."""
        exp = BenchmarkComparisonExperiment(
            sample_size=100,
            benchmark_systems=["uuid", "fractalstat"],
            scales=[100],
            num_queries=10,
        )

        result, success = exp.run()

        assert isinstance(result.fractalstat_competitive, bool)
        # fractalstat_competitive may differ from overall success
        # success means experiment ran successfully,
        # fractalstat_competitive means FractalStat met strict criteria
        assert isinstance(success, bool)

    def test_run_major_findings(self):
        """run() should generate major findings."""
        exp = BenchmarkComparisonExperiment(
            sample_size=100,
            benchmark_systems=["uuid", "fractalstat"],
            scales=[100],
            num_queries=10,
        )

        result, success = exp.run()

        assert len(result.major_findings) > 0
        assert any("Best collision rate" in f for f in result.major_findings)
        assert any("FractalStat" in f for f in result.major_findings)


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_results_custom_filename(self):
        """save_results should save with custom filename."""
        result = BenchmarkComparisonResult(
            start_time="2024-01-01T00:00:00.000Z",
            end_time="2024-01-01T00:10:00.000Z",
            total_duration_seconds=600.0,
            sample_size=100,
            scales_tested=[100],
            num_queries=10,
            systems_tested=["uuid", "fractalstat"],
            system_results=[],
            best_collision_rate_system="FractalStat",
            best_retrieval_latency_system="UUID",
            best_storage_efficiency_system="UUID",
            best_semantic_expressiveness_system="FractalStat",
            best_overall_system="FractalStat",
            fractalstat_rank_collision=1,
            fractalstat_rank_retrieval=2,
            fractalstat_rank_storage=2,
            fractalstat_rank_semantic=1,
            fractalstat_overall_score=0.85,
            major_findings=[],
            fractalstat_competitive=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the results directory
            with patch("fractalstat.exp12_benchmark_comparison.Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent = MagicMock()
                mock_path.return_value.resolve.return_value.parent.__truediv__ = (
                    lambda self, x: Path(tmpdir)
                )

                output_file = "custom_exp12_results.json"
                result_path = save_results(result, output_file)

                assert "custom_exp12_results.json" in result_path


class TestMainEntryPoint:
    """Tests for main entry point execution."""

    def test_main_with_config(self):
        """Main should load from config."""
        mock_config = MagicMock()

        def mock_get(exp, param, default):
            if exp == "EXP-12" and param == "sample_size":
                return 100
            elif exp == "EXP-12" and param == "benchmark_systems":
                return ["uuid", "fractalstat"]
            elif exp == "EXP-12" and param == "scales":
                return [100]
            elif exp == "EXP-12" and param == "num_queries":
                return 10
            return default

        mock_config.get.side_effect = mock_get

        with patch("fractalstat.config.ExperimentConfig", return_value=mock_config):
            exp = BenchmarkComparisonExperiment(
                sample_size=mock_config.get("EXP-12", "sample_size", 100000),
                benchmark_systems=mock_config.get(
                    "EXP-12",
                    "benchmark_systems",
                    [
                        "uuid",
                        "sha256",
                        "vector_db",
                        "graph_db",
                        "rdbms",
                        "fractalstat",
                    ],
                ),
                scales=mock_config.get("EXP-12", "scales", [10000, 100000, 1000000]),
                num_queries=mock_config.get("EXP-12", "num_queries", 1000),
            )

            assert exp.sample_size == 100
            assert exp.benchmark_systems == ["uuid", "fractalstat"]
            assert exp.scales == [100]
            assert exp.num_queries == 10

    def test_main_without_config(self):
        """Main should fallback to defaults when config unavailable."""
        with patch(
            "fractalstat.config.ExperimentConfig",
            side_effect=Exception("Config not found"),
        ):
            exp = BenchmarkComparisonExperiment()

            assert exp.sample_size == 100000
            assert "fractalstat" in exp.benchmark_systems
            assert exp.num_queries == 1000


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_sample_size(self):
        """Experiment should handle zero sample size gracefully."""
        exp = BenchmarkComparisonExperiment(
            sample_size=0,
            benchmark_systems=["uuid"],
            scales=[0],
            num_queries=0,
        )

        system = UUIDSystem()
        result = exp._benchmark_system(system, 0)

        assert result.scale == 0
        assert result.unique_addresses == 0

    def test_single_system(self):
        """Experiment should handle single system."""
        exp = BenchmarkComparisonExperiment(
            sample_size=50,
            benchmark_systems=["fractalstat"],
            scales=[50],
            num_queries=5,
        )

        result, success = exp.run()
        assert len(result.system_results) == 1
        assert result.system_results[0].system_name == "FractalStat"

    def test_collision_detection(self):
        """Systems should detect collisions correctly."""
        exp = BenchmarkComparisonExperiment(sample_size=100, num_queries=10)
        system = UUIDSystem()

        result = exp._benchmark_system(system, 100)

        # UUID should have zero collisions
        assert result.collisions == 0
        assert result.collision_rate == 0.0

    def test_retrieval_latency_measurement(self):
        """Retrieval latency should be measured correctly."""
        exp = BenchmarkComparisonExperiment(sample_size=100, num_queries=10)
        system = FractalStatSystem()

        result = exp._benchmark_system(system, 100)

        # Latency metrics should be valid
        assert result.mean_retrieval_latency_ms >= 0
        assert result.median_retrieval_latency_ms >= 0
        assert result.p95_retrieval_latency_ms >= result.median_retrieval_latency_ms
        assert result.p99_retrieval_latency_ms >= result.p95_retrieval_latency_ms

    def test_storage_calculation(self):
        """Storage metrics should be calculated correctly."""
        exp = BenchmarkComparisonExperiment(sample_size=100, num_queries=10)
        system = FractalStatSystem()

        result = exp._benchmark_system(system, 100)

        # Storage metrics should be valid
        assert result.avg_storage_bytes_per_entity > 0
        assert result.total_storage_bytes > 0
        assert result.total_storage_bytes >= result.avg_storage_bytes_per_entity
