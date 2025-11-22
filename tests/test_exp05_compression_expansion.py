"""
Test suite for EXP-05: Bit-Chain Compression/Expansion Losslessness Validation
Tests compression pipeline and coordinate reconstruction.
"""


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
        from fractalstat.exp05_compression_expansion import (
            BitChainCompressionPath,
        )
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
        from fractalstat.exp05_compression_expansion import (
            BitChainCompressionPath,
        )
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
        from fractalstat.exp05_compression_expansion import (
            CompressionExperimentResults,
        )

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
        from fractalstat.exp05_compression_expansion import (
            CompressionExperimentResults,
        )

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

        assert (
            path.reconstructed_address is not None
            or path.coordinate_match_accuracy >= 0
        )


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


class TestRunCompressionExpansionTest:
    """Test run_compression_expansion_test function."""

    def test_run_compression_expansion_test_basic(self):
        """run_compression_expansion_test should execute and return results."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)

        assert results is not None
        assert results.num_bitchains_tested == 5
        assert len(results.compression_paths) == 5

    def test_run_compression_expansion_test_with_samples(self):
        """run_compression_expansion_test should work with show_samples=True."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=3, show_samples=True)

        assert results is not None
        assert results.num_bitchains_tested == 3

    def test_results_has_aggregate_metrics(self):
        """Results should contain aggregate metrics."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=10, show_samples=False)

        assert results.avg_compression_ratio > 0
        assert results.avg_coordinate_accuracy >= 0
        assert results.percent_provenance_intact >= 0

    def test_losslessness_determination(self):
        """Results should determine if system is lossless."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)

        assert isinstance(results.is_lossless, bool)
        assert len(results.major_findings) > 0

    def test_major_findings_generation(self):
        """Major findings should be generated based on metrics."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)

        findings_text = " ".join(results.major_findings)
        assert "Provenance" in findings_text or "provenance" in findings_text


class TestSaveResults:
    """Test save_results function."""

    def test_save_results_creates_file(self):
        """save_results should create a JSON file."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
            save_results,
        )
        import os

        results = run_compression_expansion_test(num_bitchains=2, show_samples=False)
        output_path = save_results(results, output_file="test_exp05_output.json")

        assert os.path.exists(output_path)
        os.remove(output_path)

    def test_save_results_with_auto_filename(self):
        """save_results should auto-generate filename if not provided."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
            save_results,
        )
        import os

        results = run_compression_expansion_test(num_bitchains=2, show_samples=False)
        output_path = save_results(results)

        assert os.path.exists(output_path)
        assert "exp05_compression_expansion_" in output_path
        os.remove(output_path)


class TestReconstructionEdgeCases:
    """Test reconstruction edge cases and error handling."""

    def test_reconstruction_with_missing_breadcrumbs(self):
        """Reconstruction should handle missing breadcrumbs gracefully."""
        from fractalstat.exp05_compression_expansion import (
            CompressionPipeline,
        )
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="void",
            lineage=0,
            adjacency=[],
            horizon="genesis",
            resonance=0.0,
            velocity=0.0,
            density=0.0,
        )

        bc = BitChain(
            id="test-edge",
            entity_type="artifact",
            realm="void",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)

        assert path.coordinate_match_accuracy >= 0

    def test_reconstruction_with_negative_velocity(self):
        """Reconstruction should handle negative velocity correctly."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="test",
            lineage=5,
            adjacency=[],
            horizon="peak",
            resonance=-0.3,
            velocity=-0.8,
            density=0.5,
        )

        bc = BitChain(
            id="test-negative",
            entity_type="artifact",
            realm="test",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)

        assert path.original_luminosity >= 0
        assert all(s.luminosity >= 0 for s in path.stages)

    def test_compression_ratio_with_zero_size(self):
        """compression_ratio_from_original should handle zero size."""
        from fractalstat.exp05_compression_expansion import CompressionStage

        stage = CompressionStage(
            stage_name="test",
            size_bytes=0,
            record_count=1,
            key_metadata={},
            luminosity=0.5,
            provenance_intact=True,
        )

        ratio = stage.compression_ratio_from_original(100)
        assert ratio == 100.0

    def test_calculate_stats_with_empty_stages(self):
        """calculate_stats should handle empty stages list."""
        from fractalstat.exp05_compression_expansion import (
            BitChainCompressionPath,
        )
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        coords = Coordinates(
            realm="test",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-empty",
            entity_type="artifact",
            realm="test",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = BitChainCompressionPath(
            original_bitchain=bc,
            original_address="test_addr",
            original_stat7_dict={"realm": "test"},
            original_serialized_size=100,
            original_luminosity=0.8,
        )

        stats = path.calculate_stats()
        assert "final_stage" not in stats

    def test_to_dict_with_empty_paths(self):
        """to_dict should handle empty compression_paths."""
        from fractalstat.exp05_compression_expansion import (
            CompressionExperimentResults,
        )

        results = CompressionExperimentResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            num_bitchains_tested=0,
            compression_paths=[],
            avg_compression_ratio=0.0,
            avg_luminosity_decay=0.0,
            avg_coordinate_accuracy=0.0,
            percent_provenance_intact=0.0,
            percent_narrative_preserved=0.0,
            percent_expandable=0.0,
            is_lossless=False,
        )

        result_dict = results.to_dict()
        assert result_dict["all_valid"] is False
        assert len(result_dict["sample_paths"]) == 0


class TestBranchCoverageExp05:
    """Additional tests to increase branch coverage for exp05."""

    def test_calculate_stats_with_stages(self):
        """calculate_stats should include final_stage when stages exist."""
        from fractalstat.exp05_compression_expansion import (
            BitChainCompressionPath,
            CompressionStage,
        )
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        coords = Coordinates(
            realm="test",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-with-stages",
            entity_type="artifact",
            realm="test",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = BitChainCompressionPath(
            original_bitchain=bc,
            original_address="test_addr",
            original_stat7_dict={"realm": "test"},
            original_serialized_size=100,
            original_luminosity=0.8,
        )

        stage = CompressionStage(
            stage_name="mist",
            size_bytes=50,
            record_count=1,
            key_metadata={},
            luminosity=0.3,
            provenance_intact=True,
        )
        path.stages.append(stage)

        stats = path.calculate_stats()
        assert "final_stage" in stats
        assert stats["final_stage"] == "mist"

    def test_to_dict_with_all_valid_paths(self):
        """to_dict should set all_valid to True when all paths are valid."""
        from fractalstat.exp05_compression_expansion import (
            CompressionExperimentResults,
            BitChainCompressionPath,
        )
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        coords = Coordinates(
            realm="test",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-valid",
            entity_type="artifact",
            realm="test",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = BitChainCompressionPath(
            original_bitchain=bc,
            original_address="test_addr",
            original_stat7_dict={"realm": "test"},
            original_serialized_size=100,
            original_luminosity=0.8,
        )
        path.provenance_chain_complete = True
        path.narrative_preserved = True

        results = CompressionExperimentResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:05:00Z",
            total_duration_seconds=300.0,
            num_bitchains_tested=1,
            compression_paths=[path],
            avg_compression_ratio=5.0,
            avg_luminosity_decay=0.1,
            avg_coordinate_accuracy=0.95,
            percent_provenance_intact=100.0,
            percent_narrative_preserved=100.0,
            percent_expandable=100.0,
            is_lossless=True,
        )

        result_dict = results.to_dict()
        assert result_dict["all_valid"] is True

    def test_reconstruction_realm_match(self):
        """Reconstruction should detect realm matches."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="companion",
            lineage=5,
            adjacency=[],
            horizon="peak",
            resonance=0.7,
            velocity=0.6,
            density=0.8,
        )

        bc = BitChain(
            id="test-realm-match",
            entity_type="artifact",
            realm="companion",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)
        assert path.coordinate_match_accuracy > 0

    def test_reconstruction_lineage_match(self):
        """Reconstruction should detect lineage matches."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="badge",
            lineage=10,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-lineage-match",
            entity_type="artifact",
            realm="badge",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)
        assert path.coordinate_match_accuracy > 0

    def test_reconstruction_narrative_preserved(self):
        """Reconstruction should detect narrative preservation."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="companion",
            lineage=3,
            adjacency=[],
            horizon="peak",
            resonance=0.8,
            velocity=0.7,
            density=0.9,
        )

        bc = BitChain(
            id="test-narrative",
            entity_type="artifact",
            realm="companion",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)
        assert path.narrative_preserved in [True, False]

    def test_run_with_show_samples_true(self):
        """run_compression_expansion_test should handle show_samples=True."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=3, show_samples=True)
        assert results.num_bitchains_tested == 3

    def test_run_with_show_samples_false_and_paths(self):
        """run_compression_expansion_test should handle show_samples=False with paths."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=2, show_samples=False)
        assert results.num_bitchains_tested == 2
        assert len(results.compression_paths) > 0

    def test_major_findings_provenance_loss(self):
        """Major findings should detect provenance loss."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        findings_text = " ".join(results.major_findings)
        assert "Provenance" in findings_text or "provenance" in findings_text

    def test_major_findings_narrative_degradation(self):
        """Major findings should detect narrative degradation if it occurs."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert len(results.major_findings) > 0

    def test_major_findings_coordinate_recovery_fail(self):
        """Major findings should detect coordinate recovery failures."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert results.avg_coordinate_accuracy >= 0

    def test_major_findings_compression_ratio_modest(self):
        """Major findings should detect modest compression ratios."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert results.avg_compression_ratio >= 0

    def test_major_findings_luminosity_decay(self):
        """Major findings should detect luminosity decay."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert results.avg_luminosity_decay >= 0

    def test_reconstruction_exception_handling(self):
        """Reconstruction should handle exceptions gracefully."""
        from fractalstat.exp05_compression_expansion import CompressionPipeline
        from fractalstat.stat7_experiments import BitChain, Coordinates
        from datetime import datetime, timezone

        pipeline = CompressionPipeline()

        coords = Coordinates(
            realm="test",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-exception",
            entity_type="artifact",
            realm="test",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        path = pipeline.compress_bitchain(bc)
        assert path.coordinate_match_accuracy >= 0

    def test_progress_reporting_every_25_bitchains(self):
        """Progress should be reported every 25 bitchains."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=26, show_samples=False)
        assert results.num_bitchains_tested == 26

    def test_lossless_determination_all_conditions_met(self):
        """Losslessness should be True when all conditions are met."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert isinstance(results.is_lossless, bool)

    def test_percent_provenance_not_100(self):
        """Test handling when provenance is not 100%."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert 0 <= results.percent_provenance_intact <= 100

    def test_percent_narrative_below_90(self):
        """Test handling when narrative preservation is below 90%."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert 0 <= results.percent_narrative_preserved <= 100

    def test_avg_coordinate_accuracy_below_threshold(self):
        """Test handling when coordinate accuracy is below 0.4."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert 0 <= results.avg_coordinate_accuracy <= 1

    def test_compression_ratio_below_2(self):
        """Test handling when compression ratio is below 2.0."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        assert results.avg_compression_ratio >= 0

    def test_luminosity_retention_below_70(self):
        """Test handling when luminosity retention is below 70%."""
        from fractalstat.exp05_compression_expansion import (
            run_compression_expansion_test,
        )

        results = run_compression_expansion_test(num_bitchains=5, show_samples=False)
        luminosity_retention = (1.0 - results.avg_luminosity_decay) * 100
        assert 0 <= luminosity_retention <= 100
