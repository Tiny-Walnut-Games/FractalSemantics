"""
Embedding Provider System - Pluggable Semantic Grounding
"""

from warbler_cda.embeddings.base_provider import EmbeddingProvider
from warbler_cda.embeddings.openai_provider import OpenAIEmbeddingProvider
from warbler_cda.embeddings.local_provider import LocalEmbeddingProvider
from warbler_cda.embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from warbler_cda.embeddings.factory import EmbeddingProviderFactory

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider", 
    "LocalEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "EmbeddingProviderFactory",
]
