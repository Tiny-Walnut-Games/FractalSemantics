"""
SentenceTransformer Embedding Provider - GPU-Accelerated Semantic Grounding
High-quality embeddings using pre-trained transformer models with CUDA support
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from fractalstat.embeddings.base_provider import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerType


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """GPU-accelerated embedding provider using SentenceTransformers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        model_name_default = "all-MiniLM-L6-v2"
        self.model_name: str = (
            config.get("model_name", model_name_default)
            if config
            else model_name_default
        )
        self.batch_size: int = config.get("batch_size", 32) if config else 32
        cache_dir_default = ".embedding_cache"
        self.cache_dir: str = (
            config.get("cache_dir", cache_dir_default) if config else cache_dir_default
        )

        self.model: Optional["SentenceTransformerType"] = None
        self.device: Optional[str] = None
        self.dimension: Optional[int] = None
        self.cache: Dict[str, List[float]] = {}
        self.cache_stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "total_embeddings": 0,
        }

        self._initialize_model()
        self._load_cache()

    def _initialize_model(self) -> None:
        """Initialize the SentenceTransformer model with device detection."""
        try:
            from sentence_transformers import SentenceTransformer

            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.model is not None:
                self.dimension = self.model.get_sentence_embedding_dimension()

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        cache_key = self._get_cache_key(text)

        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            return self.cache[cache_key]

        self.cache_stats["misses"] += 1
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model first.")
        embedding: Any = self.model.encode(text, convert_to_tensor=False)

        embedding_list: List[float] = (
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        )
        self.cache[cache_key] = embedding_list
        self.cache_stats["total_embeddings"] += 1

        return embedding_list

    def embed_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and caching."""
        # Check model initialization first, before processing
        if texts and self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model first.")

        embeddings: List[List[float]] = []
        texts_to_embed: List[str] = []
        cache_keys: List[str] = []
        indices_to_embed: List[int] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_keys.append(cache_key)

            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
                self.cache_stats["hits"] += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if texts_to_embed:
            self.cache_stats["misses"] += len(texts_to_embed)
            assert self.model is not None
            batch_embeddings: Any = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                convert_to_tensor=False,
                show_progress_bar=show_progress,
            )

            for idx, batch_idx in enumerate(indices_to_embed):
                embedding: Any = batch_embeddings[idx]
                embedding_list: List[float] = (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                )
                self.cache[cache_keys[batch_idx]] = embedding_list
                self.cache_stats["total_embeddings"] += 1

            self._save_cache()

        result: List[List[float]] = []
        for cache_key in cache_keys:
            result.append(self.cache[cache_key])

        return result

    def semantic_search(
        self, query_text: str, embeddings: List[List[float]], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find k semantically similar embeddings from a list."""
        query_embedding: List[float] = self.embed_text(query_text)

        similarities: List[Tuple[int, float]] = []
        for i, emb in enumerate(embeddings):
            sim: float = self.calculate_similarity(query_embedding, emb)
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.dimension is None:
            raise RuntimeError(
                "Dimension not initialized. Call _initialize_model first."
            )
        return self.dimension

    def get_provider_info(self) -> Dict[str, Any]:
        """Get detailed provider information."""
        info = super().get_provider_info()
        info.update(
            {
                "model_name": self.model_name,
                "device": self.device,
                "batch_size": self.batch_size,
                "cache_stats": self.cache_stats.copy(),
                "cache_size": len(self.cache),
                "cache_dir": self.cache_dir,
            }
        )
        return info

    def _get_cache_key(self, text: str) -> str:
        """Generate consistent cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_cache(self) -> None:
        """Load embeddings from disk cache."""
        cache_file = Path(self.cache_dir) / (
            f"{self.model_name.replace('/', '_')}_cache.json"
        )

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    self.cache = json.load(f)
                    self.cache_stats["total_embeddings"] = len(self.cache)
            except Exception as e:
                print(f"Warning: Could not load cache from {cache_file}: {e}")

    def _save_cache(self) -> None:
        """Save embeddings to disk cache."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = Path(self.cache_dir) / (
            f"{self.model_name.replace('/', '_')}_cache.json"
        )

        try:
            with open(cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_file}: {e}")

    def compute_stat7_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """
        Compute STAT7 coordinates from embedding vector.

        Maps 384D embedding to 7D STAT7 addressing space using robust statistical
        features.
        """
        import numpy as np

        emb_array = np.array(embedding)
        dim = len(embedding)

        if dim == 0:
            return {
                "lineage": 0.5,
                "adjacency": 0.5,
                "luminosity": 0.7,
                "polarity": 0.5,
                "dimensionality": 0.5,
                "horizon": "scene",
                "realm": {"type": "semantic", "label": "embedding-derived"},
            }

        abs_emb = np.abs(emb_array)

        seg_size = dim // 7

        seg0 = emb_array[:seg_size]
        seg1 = emb_array[seg_size : 2 * seg_size]
        seg2 = emb_array[2 * seg_size : 3 * seg_size]
        seg3 = emb_array[3 * seg_size : 4 * seg_size]
        seg4 = emb_array[4 * seg_size : 5 * seg_size]
        seg5 = emb_array[5 * seg_size : 6 * seg_size]
        seg6 = emb_array[6 * seg_size :]

        lineage = float(np.mean(seg0**2))

        if len(seg1) > 1 and len(seg2) > 1 and len(seg3) > 1:

            def safe_corrcoef(a, b):
                if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                    corr = np.corrcoef(a, b)[0, 1]
                    return corr if not np.isnan(corr) else 0.0
                return 0.0

            corr_12 = safe_corrcoef(seg1, seg2)
            corr_23 = safe_corrcoef(seg2, seg3)
            corr_34 = safe_corrcoef(seg3, seg4) if len(seg4) > 1 else 0.0
            # Only compute corr_56 if both segments have the same length
            if len(seg5) > 1 and len(seg6) > 1 and len(seg5) == len(seg6):
                corr_56 = safe_corrcoef(seg5, seg6)
            else:
                corr_56 = 0.0

            adjacency = float(abs(corr_12 + corr_23 + corr_34 + corr_56) / 4.0)
        else:
            adjacency = 0.0

        luminosity = float(np.max(abs_emb[2 * seg_size : 3 * seg_size]))

        polarity_part1 = float(np.mean(seg3 > np.median(emb_array)))
        polarity_part2 = float(np.mean(seg5 > np.median(emb_array)) / 2.0)
        polarity = polarity_part1 + polarity_part2

        chunk_size = 12
        num_chunks = dim // chunk_size
        if num_chunks > 0:
            chunk_sums = [
                np.sum(abs_emb[i * chunk_size : (i + 1) * chunk_size])
                for i in range(num_chunks)
            ]
            chunk_entropy = float(np.std(chunk_sums) / (np.mean(chunk_sums) + 1e-8))
        else:
            chunk_entropy = 0.0

        high_magnitude = float(np.sum(abs_emb > np.percentile(abs_emb, 75)) / dim)
        dimensionality = (high_magnitude + min(1.0, chunk_entropy * 0.2)) / 2.0

        # Hybrid normalization preserving fractal structure:
        # - Fractal dimensions (lineage, dimensionality): unbounded, preserve scale
        # - Relational dimensions (adjacency, polarity): symmetric [-1, 1]
        # - Intensity dimensions (luminosity): asymmetric [0, 1]
        adjacency = float(np.clip(adjacency * 2.0 - 1.0, -1.0, 1.0))
        luminosity = float(np.clip(luminosity, 0.0, 1.0))
        polarity = float(np.clip(polarity * 2.0 - 1.0, -1.0, 1.0))

        return {
            "lineage": lineage,
            "adjacency": adjacency,
            "luminosity": luminosity,
            "polarity": polarity,
            "dimensionality": dimensionality,
            "horizon": "scene",
            "realm": {"type": "semantic", "label": "embedding-derived"},
        }
