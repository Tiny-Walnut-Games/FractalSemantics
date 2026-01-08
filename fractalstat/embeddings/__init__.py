"""
Embedding Provider System - Pluggable Semantic Grounding
"""

from fractalstat.embeddings.base_provider import EmbeddingProvider
from fractalstat.embeddings.openai_provider import OpenAIEmbeddingProvider
from fractalstat.embeddings.local_provider import LocalEmbeddingProvider
from fractalstat.embeddings.factory import EmbeddingProviderFactory


# Conditionally import PyTorch-dependent providers
try:
    import torch
    # Verify PyTorch can actually be used (not just imported)
    torch.tensor([1.0, 2.0])
    from fractalstat.embeddings.sentence_transformer_provider import (
        SentenceTransformerEmbeddingProvider,
    )
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    SentenceTransformerEmbeddingProvider = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMER_AVAILABLE = False


def is_sentence_transformer_available() -> bool:
    """Check if SentenceTransformerEmbeddingProvider is available."""
    return _SENTENCE_TRANSFORMER_AVAILABLE


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "LocalEmbeddingProvider",
    "EmbeddingProviderFactory",
    "is_sentence_transformer_available",
]

# Only add SentenceTransformerEmbeddingProvider if available
if _SENTENCE_TRANSFORMER_AVAILABLE:
    __all__.append("SentenceTransformerEmbeddingProvider")
