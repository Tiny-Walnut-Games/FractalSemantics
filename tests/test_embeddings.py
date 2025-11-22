"""
Comprehensive tests for embedding providers (EmbeddingProvider and subclasses)
"""

import pytest
import math


class TestEmbeddingProviderBase:
    """Test EmbeddingProvider abstract base class directly."""

    def test_embedding_provider_base_init_no_config(self):
        """EmbeddingProvider should initialize without config."""

        # Can't instantiate abstract class directly, so we'll use a concrete subclass
        from fractalstat.embeddings.local_provider import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()

        assert isinstance(provider.config, dict)
        assert len(provider.config) == 0
        assert provider.provider_id == "LocalEmbeddingProvider"
        assert isinstance(provider.created_at, float)
        assert provider.created_at > 0

    def test_embedding_provider_base_init_with_config(self):
        """EmbeddingProvider should initialize with config."""

        # Test through concrete subclass
        from fractalstat.embeddings.local_provider import LocalEmbeddingProvider

        config = {"dimension": 256, "model": "test"}
        provider = LocalEmbeddingProvider(config)

        assert provider.config == config
        assert provider.provider_id == "LocalEmbeddingProvider"

    def test_embedding_provider_abstract_methods(self):
        """EmbeddingProvider should define abstract methods."""
        from abc import ABC
        from fractalstat.embeddings.base_provider import EmbeddingProvider

        # Check that abstract methods exist in the class definition
        assert hasattr(EmbeddingProvider, "embed_text")
        assert hasattr(EmbeddingProvider, "embed_batch")
        assert hasattr(EmbeddingProvider, "get_dimension")

        # Check that they are abstract by checking the base class
        assert issubclass(EmbeddingProvider, ABC)
        assert hasattr(EmbeddingProvider, "__abstractmethods__")

    def test_calculate_similarity_method_exists(self):
        """calculate_similarity should be implemented in base class."""
        from fractalstat.embeddings.base_provider import EmbeddingProvider

        assert hasattr(EmbeddingProvider, "calculate_similarity")
        assert callable(getattr(EmbeddingProvider, "calculate_similarity"))

    def test_get_provider_info_method_exists(self):
        """get_provider_info should be implemented in base class."""
        from fractalstat.embeddings.base_provider import EmbeddingProvider

        assert hasattr(EmbeddingProvider, "get_provider_info")
        assert callable(getattr(EmbeddingProvider, "get_provider_info"))


class TestEmbeddingProvider:
    """Test base EmbeddingProvider abstract class."""

    def test_base_provider_init_with_config(self):
        """EmbeddingProvider should initialize with config."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        config = {"dimension": 256, "custom_key": "value"}
        provider = LocalEmbeddingProvider(config)

        assert provider.config == config
        assert provider.provider_id == "LocalEmbeddingProvider"
        assert isinstance(provider.created_at, float)
        assert provider.created_at > 0

    def test_base_provider_init_without_config(self):
        """EmbeddingProvider should handle no config gracefully."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        assert provider.config == {}
        assert provider.provider_id == "LocalEmbeddingProvider"

    def test_calculate_similarity_identical_vectors(self):
        """calculate_similarity should return 1.0 for identical vectors."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        vec = [1.0, 0.0, 0.0]

        similarity = provider.calculate_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_calculate_similarity_orthogonal_vectors(self):
        """calculate_similarity should return 0.0 for orthogonal vectors."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        similarity = provider.calculate_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6

    def test_calculate_similarity_opposite_vectors(self):
        """calculate_similarity should return -1.0 for opposite vectors."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]

        similarity = provider.calculate_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_calculate_similarity_zero_vector(self):
        """calculate_similarity should handle zero vectors."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = provider.calculate_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_get_provider_info(self):
        """get_provider_info should return provider metadata."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        config = {"key1": "value1", "key2": "value2"}
        provider = LocalEmbeddingProvider(config)

        info = provider.get_provider_info()

        assert info["provider_id"] == "LocalEmbeddingProvider"
        assert isinstance(info["dimension"], int)
        assert isinstance(info["created_at"], float)
        assert set(info["config_keys"]) == {"key1", "key2"}


class TestLocalEmbeddingProvider:
    """Test LocalEmbeddingProvider implementation."""

    def test_local_provider_initialization(self):
        """LocalEmbeddingProvider should initialize with default dimension."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        assert provider.get_dimension() == 128
        assert len(provider.vocabulary) == 0
        assert provider.total_documents == 0

    def test_local_provider_custom_dimension(self):
        """LocalEmbeddingProvider should respect custom dimension."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider({"dimension": 256})
        assert provider.get_dimension() == 256

    def test_embed_text_returns_correct_dimension(self):
        """embed_text should return vector of correct dimension."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider({"dimension": 64})
        embedding = provider.embed_text("hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 64
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_normalized(self):
        """embed_text should return normalized vector."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        embedding = provider.embed_text("test text")

        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 1e-6 or magnitude < 1e-6

    def test_embed_batch_multiple_texts(self):
        """embed_batch should handle multiple texts."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        texts = ["text one", "text two", "text three"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 128 for emb in embeddings)

    def test_embed_batch_empty_list(self):
        """embed_batch should handle empty list."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        embeddings = provider.embed_batch([])

        assert embeddings == []

    def test_vocabulary_updates_with_embeddings(self):
        """Vocabulary should accumulate unique tokens."""
        from fractalstat.embeddings.local_provider import (
            LocalEmbeddingProvider,
        )

        provider = LocalEmbeddingProvider()
        provider.embed_text("hello world")
        provider.embed_text("hello python")

        assert len(provider.vocabulary) > 0
        assert "hello" in provider.vocabulary or "world" in provider.vocabulary


class TestOpenAIEmbeddingProvider:
    """Test OpenAIEmbeddingProvider implementation."""

    def test_openai_provider_initialization(self):
        """OpenAIEmbeddingProvider should initialize with default model."""
        from fractalstat.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider()
        assert provider.model == "text-embedding-ada-002"
        assert provider.dimension == 1536
        assert provider.api_key is None

    def test_openai_provider_custom_config(self):
        """OpenAIEmbeddingProvider should accept custom config."""
        from fractalstat.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        config = {
            "api_key": "test-key",
            "model": "text-embedding-3-small",
            "dimension": 512,
        }
        provider = OpenAIEmbeddingProvider(config)

        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-3-small"
        assert provider.dimension == 512

    def test_openai_embed_text_creates_mock_when_api_unavailable(self):
        """embed_text should create mock embedding when API unavailable."""
        from fractalstat.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider()
        embedding = provider.embed_text("test text")

        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_openai_embed_batch_creates_mocks(self):
        """embed_batch should create mock embeddings when API unavailable."""
        from fractalstat.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider()
        texts = ["text1", "text2", "text3"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) > 0 for emb in embeddings)

    def test_openai_get_dimension(self):
        """get_dimension should return configured dimension."""
        from fractalstat.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider({"dimension": 2048})
        assert provider.get_dimension() == 2048

    def test_openai_mock_embedding_deterministic(self):
        """Mock embeddings should be deterministic for same text."""
        from fractalstat.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider()
        emb1 = provider.embed_text("test")
        emb2 = provider.embed_text("test")

        assert emb1 == emb2


class TestSentenceTransformerEmbeddingProvider:
    """Test SentenceTransformerEmbeddingProvider implementation."""

    def test_sentence_transformer_provider_init(self):
        """SentenceTransformerEmbeddingProvider should initialize."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            assert provider.batch_size == 32
            assert provider.model_name == "all-MiniLM-L6-v2"
            assert provider.cache_dir == ".embedding_cache"
            assert isinstance(provider.cache, dict)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_custom_config(self):
        """SentenceTransformerEmbeddingProvider should accept custom config."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            config = {
                "model_name": "all-MiniLM-L12-v2",
                "batch_size": 64,
                "cache_dir": ".test_cache",
            }
            provider = SentenceTransformerEmbeddingProvider(config)

            assert provider.batch_size == 64
            assert provider.cache_dir == ".test_cache"
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_embed_text_returns_list(self):
        """embed_text should return list of floats."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            embedding = provider.embed_text("test text")

            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_cache_key_deterministic(self):
        """Cache keys should be deterministic for same text."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            key1 = provider._get_cache_key("test")
            key2 = provider._get_cache_key("test")

            assert key1 == key2
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_semantic_search_structure(self):
        """semantic_search should return list of tuples."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

            results = provider.semantic_search("query", embeddings, top_k=2)

            assert isinstance(results, list)
            assert len(results) <= 2
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_sentence_transformer_provider_info(self):
        """get_provider_info should return complete information."""
        try:
            from fractalstat.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            provider = SentenceTransformerEmbeddingProvider()
            info = provider.get_provider_info()

            assert "model_name" in info
            assert "device" in info
            assert "batch_size" in info
            assert "cache_stats" in info
            assert "cache_size" in info
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestEmbeddingProviderFactory:
    """Test EmbeddingProviderFactory."""

    def test_factory_create_local_provider(self):
        """Factory should create LocalEmbeddingProvider."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        provider = EmbeddingProviderFactory.create_provider("local")

        assert provider is not None
        assert provider.__class__.__name__ == "LocalEmbeddingProvider"

    def test_factory_create_openai_provider(self):
        """Factory should create OpenAIEmbeddingProvider."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        provider = EmbeddingProviderFactory.create_provider("openai")

        assert provider is not None
        assert provider.__class__.__name__ == "OpenAIEmbeddingProvider"

    def test_factory_create_sentence_transformer_provider(self):
        """Factory should create SentenceTransformerEmbeddingProvider."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        try:
            provider = EmbeddingProviderFactory.create_provider("sentence_transformer")
            assert provider is not None
            assert provider.__class__.__name__ == "SentenceTransformerEmbeddingProvider"
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_factory_unknown_provider_raises(self):
        """Factory should raise ValueError for unknown provider."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        with pytest.raises(ValueError, match="Unknown provider type"):
            EmbeddingProviderFactory.create_provider("unknown_provider")

    def test_factory_list_available_providers(self):
        """Factory should list all available providers."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        providers = EmbeddingProviderFactory.list_available_providers()

        assert isinstance(providers, list)
        assert "local" in providers
        assert "openai" in providers
        assert "sentence_transformer" in providers

    def test_factory_create_with_config(self):
        """Factory should pass config to provider."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        config = {"dimension": 256, "custom": "value"}
        provider = EmbeddingProviderFactory.create_provider("local", config)

        assert provider.config == config
        assert provider.get_dimension() == 256

    def test_factory_get_default_provider(self):
        """Factory should return default provider."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        provider = EmbeddingProviderFactory.get_default_provider()

        assert provider is not None
        assert hasattr(provider, "embed_text")
        assert hasattr(provider, "embed_batch")

    def test_factory_create_from_config(self):
        """Factory should create provider from config dict."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        full_config = {"provider": "local", "config": {"dimension": 512}}
        provider = EmbeddingProviderFactory.create_from_config(full_config)

        assert provider.get_dimension() == 512

    def test_factory_create_from_config_default_provider(self):
        """Factory should use local as default provider type."""
        from fractalstat.embeddings.factory import EmbeddingProviderFactory

        full_config = {"config": {"dimension": 256}}
        provider = EmbeddingProviderFactory.create_from_config(full_config)

        assert provider.__class__.__name__ == "LocalEmbeddingProvider"


class TestEmbeddingsPackageExports:
    """Test embeddings package exports."""

    def test_package_exports_all_classes(self):
        """Package should export all embedding classes."""
        from fractalstat import embeddings

        assert hasattr(embeddings, "EmbeddingProvider")
        assert hasattr(embeddings, "LocalEmbeddingProvider")
        assert hasattr(embeddings, "OpenAIEmbeddingProvider")
        assert hasattr(embeddings, "EmbeddingProviderFactory")

    def test_all_list(self):
        """Package __all__ should be complete."""
        from fractalstat.embeddings import __all__

        assert "EmbeddingProvider" in __all__
        assert "LocalEmbeddingProvider" in __all__
        assert "OpenAIEmbeddingProvider" in __all__
        assert "EmbeddingProviderFactory" in __all__
