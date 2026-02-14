"""
Embedding Provider Factory - Dynamic Provider Creation
"""

from typing import Any, Optional, dict, list

from fractalsemantics.embeddings.base_provider import EmbeddingProvider
from fractalsemantics.embeddings.local_provider import LocalEmbeddingProvider
from fractalsemantics.embeddings.openai_provider import OpenAIEmbeddingProvider

# Import sentence transformer provider conditionally
try:
    from fractalsemantics.embeddings.sentence_transformer_provider import (
        SentenceTransformerEmbeddingProvider,
    )
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    SentenceTransformerEmbeddingProvider = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMER_AVAILABLE = False


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    PROVIDERS = {
        "local": LocalEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
    }

    # Add sentence transformer provider if available
    if _SENTENCE_TRANSFORMER_AVAILABLE:
        PROVIDERS["sentence_transformer"] = SentenceTransformerEmbeddingProvider

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        config: Optional[dict[str, any]] = None,
    ) -> EmbeddingProvider:
        """Create an embedding provider of the specified type."""
        if provider_type not in cls.PROVIDERS:
            available = list(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider type '{provider_type}'. Available: {available}"
            )

        provider_class = cls.PROVIDERS[provider_type]
        return provider_class(config)  # type: ignore

    @classmethod
    def get_default_provider(
        cls, config: Optional[dict[str, any]] = None
    ) -> EmbeddingProvider:
        """Get the default embedding provider (SentenceTransformer with fallback)."""
        if "sentence_transformer" in cls.PROVIDERS:
            try:
                return cls.create_provider("sentence_transformer", config)
            except Exception:
                print(
                    "Warning: SentenceTransformer failed to initialize, "
                    "falling back to LocalEmbeddingProvider"
                )
                return cls.create_provider("local", config)
        else:
            print(
                "Warning: SentenceTransformer not available, "
                "using LocalEmbeddingProvider"
            )
            return cls.create_provider("local", config)

    @classmethod
    def list_available_providers(cls) -> list[str]:
        """list all available provider types."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def create_from_config(cls, full_config: dict[str, any]) -> EmbeddingProvider:
        """Create provider from configuration dict."""
        provider_type = full_config.get("provider", "local")
        provider_config = full_config.get("config", {})

        return cls.create_provider(provider_type, provider_config)
