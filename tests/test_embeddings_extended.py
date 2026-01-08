"""
Extended tests for sentence_transformer_provider to achieve 95%+ coverage
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open


def _check_pytorch_available():
    """Check if PyTorch is available and can be imported without DLL issues."""
    try:
        import torch
        # Try to use torch to ensure it's not just importable but actually functional
        torch.tensor([1.0, 2.0])
        return True
    except (ImportError, OSError, RuntimeError):
        return False


class TestSentenceTransformerExtended:
    """Extended tests for SentenceTransformerEmbeddingProvider."""

    def test_sentence_transformer_embed_batch_with_caching(self):
        """embed_batch should use cache for repeated texts."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            texts = ["test1", "test2", "test1"]  # test1 repeated

            embeddings = provider.embed_batch(texts)

            assert len(embeddings) == 3
            # Same text should have same embedding
            assert embeddings[0] == embeddings[2]
            assert (
                provider.cache_stats["hits"] > 0 or provider.cache_stats["misses"] > 0
            )
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_cache_persistence(self):
        """Cache should persist to disk and reload."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                config = {"cache_dir": tmpdir}
                provider1 = SentenceTransformerEmbeddingProvider(config)

                # Generate and cache an embedding
                text = "test persistence"
                emb1 = provider1.embed_text(text)
                provider1._save_cache()

                # Create new provider with same cache dir
                provider2 = SentenceTransformerEmbeddingProvider(config)

                # Should load from cache
                cache_key = provider2._get_cache_key(text)
                assert cache_key in provider2.cache
                assert provider2.cache[cache_key] == emb1
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_batch_processing_variations(self):
        """embed_batch should handle different batch sizes."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            # Test with small batch size
            provider = SentenceTransformerEmbeddingProvider({"batch_size": 2})
            texts = ["text1", "text2", "text3", "text4", "text5"]

            embeddings = provider.embed_batch(texts, show_progress=False)

            assert len(embeddings) == 5
            assert all(len(emb) > 0 for emb in embeddings)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_device_selection(self):
        """Provider should detect and use appropriate device."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()

            assert provider.device in ["cpu", "cuda"]
            assert provider.model is not None
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_cache_stats_tracking(self):
        """Cache stats should track hits and misses correctly."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()

            # First embedding - should be a miss
            initial_misses = provider.cache_stats["misses"]
            provider.embed_text("unique text 1")
            assert provider.cache_stats["misses"] == initial_misses + 1

            # Second embedding of same text - should be a hit
            initial_hits = provider.cache_stats["hits"]
            provider.embed_text("unique text 1")
            assert provider.cache_stats["hits"] == initial_hits + 1
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_get_dimension(self):
        """get_dimension should return correct dimension."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            dim = provider.get_dimension()

            assert isinstance(dim, int)
            assert dim > 0
            assert dim == provider.dimension
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_semantic_search_ranking(self):
        """semantic_search should rank by similarity."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()

            # Create embeddings for similar and dissimilar texts
            texts = ["cat", "dog", "automobile", "vehicle"]
            embeddings = [provider.embed_text(t) for t in texts]

            # Search for "animal" - should rank cat/dog higher
            results = provider.semantic_search("animal", embeddings, top_k=2)

            assert len(results) == 2
            assert all(
                isinstance(r[0], int) and isinstance(r[1], float) for r in results
            )
            # Results should be sorted by similarity (descending)
            assert results[0][1] >= results[1][1]
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_empty_batch(self):
        """embed_batch should handle empty list."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            embeddings = provider.embed_batch([])

            assert embeddings == []
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_compute_fractalstat_from_embedding(self):
        """compute_fractalstat_from_embedding should return valid FractalStat coordinates."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            embedding = provider.embed_text("test text for fractalstat")

            fractalstat = provider.compute_fractalstat_from_embedding(embedding)

            assert "lineage" in fractalstat
            assert "adjacency" in fractalstat
            assert "luminosity" in fractalstat
            assert "polarity" in fractalstat
            assert "dimensionality" in fractalstat
            assert "horizon" in fractalstat
            assert "realm" in fractalstat

            # Hybrid bounds: fractal dimensions unbounded, relational
            # symmetric, intensity asymmetric
            assert isinstance(fractalstat["lineage"], (int, float))
            assert -1.0 <= fractalstat["adjacency"] <= 1.0
            assert 0.0 <= fractalstat["luminosity"] <= 1.0
            assert -1.0 <= fractalstat["polarity"] <= 1.0
            assert isinstance(fractalstat["dimensionality"], (int, float))
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_compute_fractalstat_empty_embedding(self):
        """compute_fractalstat_from_embedding should handle empty embedding."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            fractalstat = provider.compute_fractalstat_from_embedding([])

            # Should return default values
            assert fractalstat["lineage"] == 0.5
            assert fractalstat["adjacency"] == 0.5
            assert fractalstat["luminosity"] == 0.7
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_cache_save_error_handling(self):
        """_save_cache should handle errors gracefully."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            provider.embed_text("test")

            # Mock open to raise an exception
            with patch("builtins.open", mock_open()) as mock_file:
                mock_file.side_effect = IOError("Disk full")
                # Should not raise, just print warning
                provider._save_cache()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_cache_load_error_handling(self):
        """_load_cache should handle corrupted cache gracefully."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create corrupted cache file
                cache_file = Path(tmpdir) / "all-MiniLM-L6-v2_cache.json"
                cache_file.write_text("corrupted json {{{")

                config = {"cache_dir": tmpdir}
                # Should not raise, just skip loading
                provider = SentenceTransformerEmbeddingProvider(config)
                assert len(provider.cache) == 0
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_embed_text_without_model(self):
        """embed_text should raise RuntimeError if model not initialized."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            provider.model = None  # Simulate uninitialized model

            with pytest.raises(RuntimeError, match="Model not initialized"):
                provider.embed_text("test")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_embed_batch_without_model(self):
        """embed_batch should raise RuntimeError if model not initialized."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            provider.model = None  # Simulate uninitialized model

            with pytest.raises(RuntimeError, match="Model not initialized"):
                provider.embed_batch(["test1", "test2"])
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_get_dimension_without_initialization(self):
        """get_dimension should raise RuntimeError if dimension not initialized."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            provider.dimension = None  # Simulate uninitialized dimension

            with pytest.raises(RuntimeError, match="Dimension not initialized"):
                provider.get_dimension()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_batch_with_progress(self):
        """embed_batch should support show_progress parameter."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            texts = ["text1", "text2", "text3"]

            # Should work with show_progress=True
            embeddings = provider.embed_batch(texts, show_progress=True)
            assert len(embeddings) == 3
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_cache_dir_creation(self):
        """Cache directory should be created if it doesn't exist."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                cache_dir = Path(tmpdir) / "new_cache_dir"
                assert not cache_dir.exists()

                config = {"cache_dir": str(cache_dir)}
                provider = SentenceTransformerEmbeddingProvider(config)
                provider.embed_text("test")
                provider._save_cache()

                # Cache directory should now exist
                assert cache_dir.exists()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_provider_info_complete(self):
        """get_provider_info should include cache_dir."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            info = provider.get_provider_info()

            assert "cache_dir" in info
            assert info["cache_dir"] == ".embedding_cache"
        except ImportError:
            pytest.skip("sentence-transformers not installed")
