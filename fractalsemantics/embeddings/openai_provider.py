"""
OpenAI Embedding Provider - Cloud-based Semantic Grounding
"""

import hashlib
import math
import struct
from typing import Any, Optional, TypeAlias

from fractalsemantics.embeddings.base_provider import EmbeddingProvider

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.api_key: Optional[str] = config.get("api_key") if config else None
        model_default = "text-embedding-3-small"
        self.model: str = (
            config.get("model", model_default) if config else model_default
        )
        self.dimension: int = config.get("dimension", 1536) if config else 1536
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai

                if hasattr(openai, "OpenAI"):
                    if self.api_key:
                        self._client = openai.OpenAI(api_key=self.api_key)
                    else:
                        self._client = openai.OpenAI()
                else:
                    if self.api_key:
                        openai.api_key = self.api_key
                    self._client = openai
            except ImportError as exc:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                ) from exc
        return self._client

    def _extract_embeddings(self, response: Any) -> list[list[float]]:
        """Extract embedding vectors from both modern and legacy OpenAI responses."""
        if isinstance(response, dict):
            data = response.get("data", [])
            return [list(item["embedding"]) for item in data if "embedding" in item]

        data = getattr(response, "data", None)
        if data is None:
            return []

        embeddings: list[list[float]] = []
        for item in data:
            value = getattr(item, "embedding", None)
            if value is None and isinstance(item, dict):
                value = item.get("embedding")
            if value is not None:
                embeddings.append(list(value))
        return embeddings

    def _request_embeddings(self, input_payload: Any) -> list[list[float]]:
        """Request embeddings with SDK-version compatible API calls."""
        client = self._get_client()

        if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
            response = client.embeddings.create(model=self.model, input=input_payload)
            return self._extract_embeddings(response)

        if hasattr(client, "Embedding") and hasattr(client.Embedding, "create"):
            response = client.Embedding.create(model=self.model, input=input_payload)  # pylint: disable=no-member
            return self._extract_embeddings(response)

        raise RuntimeError("Unsupported OpenAI client interface for embeddings")

    def embed_text(self, text: str) -> list[float]:
        """Generate OpenAI embedding for text."""
        try:
            embeddings = self._request_embeddings(text)
            if embeddings:
                return embeddings[0]
            raise RuntimeError("Empty embedding response")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Warning: OpenAI API failed ({e}), using mock embedding")
            return self._create_mock_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate OpenAI embeddings for multiple texts."""
        try:
            embeddings = self._request_embeddings(texts)
            if embeddings:
                return embeddings
            raise RuntimeError("Empty embedding response")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Warning: OpenAI API failed ({e}), using mock embeddings")
            return [self._create_mock_embedding(text) for text in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def _create_mock_embedding(self, text: str) -> list[float]:
        """Create a mock embedding for development/testing."""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        vector = []
        for i in range(0, min(len(hash_bytes), self.dimension // 4 * 4), 4):
            value = struct.unpack("f", hash_bytes[i : i + 4])[0]
            vector.append(value)

        while len(vector) < self.dimension:
            seed = len(vector) + hash(text)
            normalized_val = (seed % 1000) / 1000.0 - 0.5
            vector.append(normalized_val)

        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector
