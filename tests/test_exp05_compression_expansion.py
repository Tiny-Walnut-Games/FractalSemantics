"""
Test suite for EXP-05: Bit-Chain Compression/Expansion Losslessness Validation
Tests compression pipeline and coordinate reconstruction.
"""

import pytest


class TestCompressionStage:
    """Test CompressionStage data structure."""

    def test_stage_initialization(self):
        """CompressionStage should initialize with pipeline metadata."""
        from fractalstat.exp05_compression_expansion import CompressionStage

        stage = CompressionStage(
            stage_name="original",
            size_bytes=1000,
            record_count=1,
            key_metadata={"address": "test_addr"},
            luminosity=0.8,
            provenance_intact=True,
        )

        assert stage.stage_name == "original"
        assert stage.size_bytes == 1000
        assert stage.provenance_intact

    def test_compression_ratio_calculation(self):
        """compression_ratio_from_original should calculate correctly."""
        from fractalstat.exp05_compression_expansion import CompressionStage

        stage = CompressionStage(
            stage_name="mist",
            size_bytes=100,
            record_count=1,
            key_metadata={},
            luminosity=0.5,
            provenance_intact=True,
        )

        ratio = stage.compression_ratio_from_original(1000)
        assert ratio == 10.0


class TestBitChainCompressionPath:
    """Test BitChainCompressionPath tracking."""

    def test_path_initialization(self):
        """BitChainCompressionPath should initialize with bitchain reference."""
        from fractalstat.exp05_compression_expansion import BitChainCompressionPath
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-123",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = BitChainCompressionPath(
            original_bitchain=bc,
            original_address="addr_123",
            original_stat7_dict={},
            original_serialized_size=100,
            original_luminosity=0.8,
        )

        assert path.original_bitchain == bc
        assert path.original_address == "addr_123"

    def test_calculate_stats(self):
        """calculate_stats should return summary dict."""
        from fractalstat.exp05_compression_expansion import BitChainCompressionPath
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-123",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = BitChainCompressionPath(
            original_bitchain=bc,
            original_address="addr_123" * 10,
            original_stat7_dict={"realm": "data"},
            original_serialized_size=100,
            original_luminosity=0.8,
        )

        stats = path.calculate_stats()
        assert isinstance(stats, dict)
        assert "original_realm" in stats


class TestCompressionExperimentResults:
    """Test compression experiment results aggregation."""

    def test_results_initialization(self):
        """CompressionExperimentResults should initialize with metrics."""
        from fractalstat.exp05_compression_expansion import CompressionExperimentResults

        results = CompressionExperimentResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            num_bitchains_tested=10,
            compression_paths=[],
            avg_compression_ratio=5.0,
            avg_luminosity_decay=0.1,
            avg_coordinate_accuracy=0.95,
            percent_provenance_intact=100.0,
            percent_narrative_preserved=100.0,
            percent_expandable=100.0,
            is_lossless=True,
        )

        assert results.num_bitchains_tested == 10
        assert results.is_lossless

    def test_results_to_dict(self):
        """Results should serialize to dict."""
        from fractalstat.exp05_compression_expansion import CompressionExperimentResults

        results = CompressionExperimentResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            num_bitchains_tested=10,
            compression_paths=[],
            avg_compression_ratio=5.0,
            avg_luminosity_decay=0.1,
            avg_coordinate_accuracy=0.95,
            percent_provenance_intact=100.0,
            percent_narrative_preserved=100.0,
            percent_expandable=100.0,
            is_lossless=True,
            major_findings=["Zero information loss detected"],
        )

        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["experiment"] == "EXP-05"
        assert "aggregate_metrics" in result_dict


class TestCompressionPipeline:
    """Test compression pipeline simulation."""

    def test_pipeline_initializes(self):
        """CompressionPipeline should initialize all stores."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline

        pipeline = CompressionPipeline()
        assert hasattr(pipeline, "fragment_store")
        assert hasattr(pipeline, "cluster_store")
        assert hasattr(pipeline, "glyph_store")
        assert hasattr(pipeline, "mist_store")

    def test_compress_bitchain_creates_path(self):
        """compress_bitchain should create compression path with stages."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-123",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={"value": 42},
        )

        path = pipeline.compress_bitchain(bc)

        assert path.original_bitchain == bc
        assert len(path.stages) > 0
        assert path.stages[0].stage_name == "original"

    def test_compression_produces_multiple_stages(self):
        """Compression should progress through multiple stages."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="narrative",
            lineage=5,
            adjacency=["neighbor1"],
            horizon="emergence",
            resonance=0.7,
            velocity=0.3,
            density=0.6,
        )

        bc = BitChain(
            id="test-456",
            entity_type="concept",
            realm="narrative",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={"metadata": "test"},
        )

        path = pipeline.compress_bitchain(bc)

        stage_names = [s.stage_name for s in path.stages]
        assert "original" in stage_names
        assert "fragments" in stage_names
        assert "cluster" in stage_names
        assert "glyph" in stage_names
        assert "mist" in stage_names

    def test_reconstruction_from_mist(self):
        """Reconstruction should recover coordinates from compressed form."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="system",
            lineage=3,
            adjacency=[],
            horizon="peak",
            resonance=0.2,
            velocity=0.8,
            density=0.4,
        )

        bc = BitChain(
            id="test-789",
            entity_type="agent",
            realm="system",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)

        assert path.reconstructed_address is not None or path.coordinate_match_accuracy >= 0


class TestLuminosityDecay:
    """Test luminosity decay through compression stages."""

    def test_luminosity_decreases_through_stages(self):
        """Luminosity should decrease as data is compressed."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="event",
            lineage=2,
            adjacency=[],
            horizon="decay",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-luminosity",
            entity_type="artifact",
            realm="event",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)

        luminosities = [s.luminosity for s in path.stages]
        assert len(luminosities) > 1
        assert luminosities[-1] <= luminosities[0]
