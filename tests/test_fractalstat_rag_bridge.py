"""
Comprehensive tests for FractalStat-RAG Bridge
"""

import pytest


class TestRealm:
    """Test Realm data structure."""

    def test_realm_initialization(self):
        """Realm should initialize with type and label."""
        from fractalstat.fractalstat_rag_bridge import Realm

        realm = Realm(type="game", label="Main Story")

        assert realm.type == "game"
        assert realm.label == "Main Story"

    def test_realm_various_types(self):
        """Realm should support various types."""
        from fractalstat.fractalstat_rag_bridge import Realm

        types = [
            "game",
            "system",
            "faculty",
            "pattern",
            "data",
            "narrative",
            "business",
            "concept",
        ]

        for realm_type in types:
            realm = Realm(type=realm_type, label="Test")
            assert realm.type == realm_type


class TestFractalStatAddress:
    """Test FractalStatAddress data structure."""

    def test_fractalstat_address_initialization(self):
        """FractalStatAddress should initialize with all coordinates."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")
        address = FractalStatAddress(
            realm=realm,
            lineage=5,
            adjacency=0.8,
            horizon="scene",
            luminosity=0.9,
            polarity=0.7,
            dimensionality=3,
            alignment="balanced_pragmatic",
        )

        assert address.realm == realm
        assert address.lineage == 5
        assert address.adjacency == 0.8
        assert address.horizon == "scene"
        assert address.luminosity == 0.9
        assert address.polarity == 0.7
        assert address.dimensionality == 3
        assert address.alignment == "balanced_pragmatic"

    def test_fractalstat_address_validates_adjacency_range(self):
        """FractalStatAddress should validate adjacency in [0, 1]."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")

        with pytest.raises(AssertionError):
            FractalStatAddress(
                realm=realm,
                lineage=0,
                adjacency=1.5,
                horizon="scene",
                luminosity=0.5,
                polarity=0.5,
                dimensionality=3,
                alignment="flexible_pragmatic",
            )

    def test_fractalstat_address_validates_luminosity_range(self):
        """FractalStatAddress should validate luminosity in [0, 1]."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")

        with pytest.raises(AssertionError):
            FractalStatAddress(
                realm=realm,
                lineage=0,
                adjacency=0.5,
                horizon="scene",
                luminosity=1.5,
                polarity=0.5,
                dimensionality=3,
                alignment="flexible_pragmatic",
            )

    def test_fractalstat_address_validates_polarity_range(self):
        """FractalStatAddress should validate polarity in [0, 1]."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")

        with pytest.raises(AssertionError):
            FractalStatAddress(
                realm=realm,
                lineage=0,
                adjacency=0.5,
                horizon="scene",
                luminosity=0.5,
                polarity=-0.5,
                dimensionality=3,
                alignment="flexible_pragmatic",
            )

    def test_fractalstat_address_validates_lineage_nonnegative(self):
        """FractalStatAddress should validate lineage >= 0."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")

        with pytest.raises(AssertionError):
            FractalStatAddress(
                realm=realm,
                lineage=-1,
                adjacency=0.5,
                horizon="scene",
                luminosity=0.5,
                polarity=0.5,
                dimensionality=3,
                alignment="flexible_pragmatic",
            )

    def test_fractalstat_address_validates_dimensionality_range(self):
        """FractalStatAddress should validate dimensionality in [1, 7]."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")

        with pytest.raises(AssertionError):
            FractalStatAddress(
                realm=realm,
                lineage=0,
                adjacency=0.5,
                horizon="scene",
                luminosity=0.5,
                polarity=0.5,
                dimensionality=8,
                alignment="structured_destructive",
            )

    def test_fractalstat_address_to_dict(self):
        """FractalStatAddress should convert to dictionary."""
        from fractalstat.fractalstat_rag_bridge import FractalStatAddress, Realm

        realm = Realm(type="game", label="Main Story")
        address = FractalStatAddress(
            realm=realm,
            lineage=5,
            adjacency=0.8,
            horizon="scene",
            luminosity=0.9,
            polarity=0.7,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        addr_dict = address.to_dict()

        assert isinstance(addr_dict, dict)
        assert addr_dict["realm"]["type"] == "game"
        assert addr_dict["realm"]["label"] == "Main Story"
        assert addr_dict["lineage"] == 5
        assert addr_dict["adjacency"] == 0.8
        assert addr_dict["horizon"] == "scene"
        assert addr_dict["luminosity"] == 0.9
        assert addr_dict["polarity"] == 0.7
        assert addr_dict["dimensionality"] == 3
        assert addr_dict["alignment"] == "flexible_pragmatic"


class TestRAGDocument:
    """Test RAGDocument data structure."""

    def test_rag_document_initialization(self):
        """RAGDocument should initialize with all fields."""
        from fractalstat.fractalstat_rag_bridge import (
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        address = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        doc = RAGDocument(
            id="doc-001",
            text="Sample document text",
            embedding=[1.0, 0.0, 0.0],
            fractalstat=address,
            metadata={"source": "test"},
        )

        assert doc.id == "doc-001"
        assert doc.text == "Sample document text"
        assert doc.embedding == [1.0, 0.0, 0.0]
        assert doc.fractalstat == address
        assert doc.metadata == {"source": "test"}

    def test_rag_document_validates_nonempty_embedding(self):
        """RAGDocument should validate non-empty embedding."""
        from fractalstat.fractalstat_rag_bridge import (
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        address = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        with pytest.raises(AssertionError):
            RAGDocument(id="doc-001", text="Sample text", embedding=[], fractalstat=address)

    def test_rag_document_default_metadata(self):
        """RAGDocument should have empty metadata by default."""
        from fractalstat.fractalstat_rag_bridge import (
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        address = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        doc = RAGDocument(
            id="doc-001", text="Sample", embedding=[1.0, 0.0], fractalstat=address
        )

        assert doc.metadata == {}


class TestCosineSimilarity:
    """Test cosine_similarity function."""

    def test_cosine_similarity_identical_vectors(self):
        """cosine_similarity should return 1.0 for identical vectors."""
        from fractalstat.fractalstat_rag_bridge import cosine_similarity

        vec = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """cosine_similarity should return ~0.0 for orthogonal vectors."""
        from fractalstat.fractalstat_rag_bridge import cosine_similarity

        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = cosine_similarity(vec1, vec2)

        assert abs(similarity) < 1e-6

    def test_cosine_similarity_opposite_vectors(self):
        """cosine_similarity should return -1.0 for opposite vectors."""
        from fractalstat.fractalstat_rag_bridge import cosine_similarity

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_empty_vectors(self):
        """cosine_similarity should handle empty vectors."""
        from fractalstat.fractalstat_rag_bridge import cosine_similarity

        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_cosine_similarity_zero_magnitude_vector(self):
        """cosine_similarity should handle zero-magnitude vectors."""
        from fractalstat.fractalstat_rag_bridge import cosine_similarity

        zero_vec = [0.0, 0.0, 0.0]
        vec = [1.0, 0.0, 0.0]

        similarity = cosine_similarity(zero_vec, vec)
        assert similarity == 0.0

    def test_cosine_similarity_normalized(self):
        """cosine_similarity should work with normalized vectors."""
        from fractalstat.fractalstat_rag_bridge import cosine_similarity

        vec1 = [0.6, 0.8]  # normalized: 0.36 + 0.64 = 1.0
        vec2 = [0.8, 0.6]

        similarity = cosine_similarity(vec1, vec2)
        expected = (0.6 * 0.8 + 0.8 * 0.6) / (1.0 * 1.0)  # 0.96

        assert abs(similarity - expected) < 1e-6


class TestFractalStatResonance:
    """Test fractalstat_resonance function."""

    def test_fractalstat_resonance_same_realm_returns_high_score(self):
        """fractalstat_resonance should return high score for same realm."""
        from fractalstat.fractalstat_rag_bridge import (
            fractalstat_resonance,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        query = FractalStatAddress(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )
        doc = FractalStatAddress(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        resonance = fractalstat_resonance(query, doc)
        assert resonance >= 0.85

    def test_fractalstat_resonance_different_realm_returns_lower_score(self):
        """fractalstat_resonance should return lower score for different realm."""
        from fractalstat.fractalstat_rag_bridge import (
            fractalstat_resonance,
            FractalStatAddress,
            Realm,
        )

        realm1 = Realm(type="game", label="Main")
        realm2 = Realm(type="system", label="Core")

        query = FractalStatAddress(
            realm=realm1,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )
        doc = FractalStatAddress(
            realm=realm2,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        resonance = fractalstat_resonance(query, doc)
        assert 0.0 <= resonance <= 1.0

    def test_fractalstat_resonance_range(self):
        """fractalstat_resonance should always return value in [0, 1]."""
        from fractalstat.fractalstat_rag_bridge import (
            fractalstat_resonance,
            FractalStatAddress,
            Realm,
        )

        realm1 = Realm(type="game", label="Main")
        realm2 = Realm(type="data", label="Records")

        addresses = []
        for lineage in [0, 5, 10]:
            for adjacency in [0.0, 0.5, 1.0]:
                addr = FractalStatAddress(
                    realm=realm1 if lineage % 2 == 0 else realm2,
                    lineage=lineage,
                    adjacency=adjacency,
                    horizon="scene",
                    luminosity=0.5 + adjacency / 2,
                    polarity=0.5,
                    dimensionality=3,
                    alignment="flexible_pragmatic",
                )
                addresses.append(addr)

        for query in addresses[:3]:
            for doc in addresses[3:]:
                resonance = fractalstat_resonance(query, doc)
                assert 0.0 <= resonance <= 1.0


class TestHybridScoring:
    """Test hybrid scoring functions."""

    def test_hybrid_score_combines_semantic_and_fractalstat(self):
        """Hybrid scoring should combine semantic similarity and FractalStat resonance."""
        from fractalstat.fractalstat_rag_bridge import (
            hybrid_score,
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        query_addr = FractalStatAddress(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )
        doc_addr = FractalStatAddress(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        query_embedding = [1.0, 0.0, 0.0]
        doc = RAGDocument(
            id="doc-1", text="test", embedding=[1.0, 0.0, 0.0], fractalstat=doc_addr
        )

        score = hybrid_score(
            query_embedding=query_embedding,
            doc=doc,
            query_fractalstat=query_addr,
            weight_semantic=0.5,
            weight_fractalstat=0.5,
        )

        assert 0.0 <= score <= 1.0

    def test_hybrid_score_different_weights(self):
        """Hybrid scoring should respect different weights."""
        from fractalstat.fractalstat_rag_bridge import (
            hybrid_score,
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        query_emb = [1.0, 0.0]
        doc = RAGDocument(id="doc-1", text="test", embedding=[1.0, 0.0], fractalstat=addr)

        score1 = hybrid_score(query_emb, doc, addr, 0.9, 0.1)
        score2 = hybrid_score(query_emb, doc, addr, 0.1, 0.9)

        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0

    def test_hybrid_score_weights_must_sum_to_one(self):
        """hybrid_score should require weights to sum to 1.0."""
        from fractalstat.fractalstat_rag_bridge import (
            hybrid_score,
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        doc = RAGDocument(id="doc-1", text="test", embedding=[1.0, 0.0], fractalstat=addr)

        with pytest.raises(AssertionError):
            hybrid_score([1.0, 0.0], doc, addr, 0.6, 0.5)


class TestRAGRetrieval:
    """Test RAG retrieval functionality."""

    def test_retrieve_returns_ranked_list(self):
        """retrieve should return ranked list of documents."""
        from fractalstat.fractalstat_rag_bridge import (
            retrieve,
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        docs = [
            RAGDocument(id="d1", text="text1", embedding=[1.0, 0.0], fractalstat=addr),
            RAGDocument(id="d2", text="text2", embedding=[0.0, 1.0], fractalstat=addr),
            RAGDocument(id="d3", text="text3", embedding=[1.0, 1.0], fractalstat=addr),
        ]

        query_emb = [1.0, 0.0]
        results = retrieve(
            documents=docs,
            query_embedding=query_emb,
            query_fractalstat=addr,
            k=2,
            weight_semantic=0.5,
            weight_fractalstat=0.5,
        )

        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_retrieve_respects_k_parameter(self):
        """retrieve should respect k parameter."""
        from fractalstat.fractalstat_rag_bridge import (
            retrieve,
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        docs = [
            RAGDocument(
                id=f"d{i}",
                text=f"text{i}",
                embedding=[float(i) / 10, 0.0],
                fractalstat=addr,
            )
            for i in range(10)
        ]

        results = retrieve(
            documents=docs,
            query_embedding=[1.0, 0.0],
            query_fractalstat=addr,
            k=3,
            weight_semantic=0.5,
            weight_fractalstat=0.5,
        )

        assert len(results) <= 3

    def test_retrieve_empty_documents_list(self):
        """retrieve should handle empty document list."""
        from fractalstat.fractalstat_rag_bridge import retrieve, FractalStatAddress, Realm

        realm = Realm(type="game", label="Main")
        addr = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        results = retrieve(
            documents=[],
            query_embedding=[1.0, 0.0],
            query_fractalstat=addr,
            k=5,
            weight_semantic=0.5,
            weight_fractalstat=0.5,
        )

        assert results == []

    def test_retrieve_semantic_only(self):
        """retrieve_semantic_only should work without FractalStat."""
        from fractalstat.fractalstat_rag_bridge import (
            retrieve_semantic_only,
            RAGDocument,
            FractalStatAddress,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = FractalStatAddress(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3,
            alignment="flexible_pragmatic",
        )

        docs = [
            RAGDocument(id="d1", text="text1", embedding=[1.0, 0.0], fractalstat=addr),
            RAGDocument(id="d2", text="text2", embedding=[0.0, 1.0], fractalstat=addr),
        ]

        results = retrieve_semantic_only(
            documents=docs, query_embedding=[1.0, 0.0], k=2
        )

        assert isinstance(results, list)
        assert len(results) <= 2


class TestSynthenticDataGeneration:
    """Test synthetic data generation for testing."""

    def test_generate_random_fractalstat_address(self):
        """generate_random_fractalstat_address should create valid address."""
        from fractalstat.fractalstat_rag_bridge import (
            generate_random_fractalstat_address,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = generate_random_fractalstat_address(realm)

        assert addr is not None
        assert addr.realm == realm
        assert hasattr(addr, "lineage")
        assert 0.0 <= addr.adjacency <= 1.0
        assert 0.0 <= addr.luminosity <= 1.0
        assert 0.0 <= addr.polarity <= 1.0
        assert addr.lineage >= 0
        assert 1 <= addr.dimensionality <= 7
        assert addr.alignment == "flexible_pragmatic"

    def test_generate_random_fractalstat_address_custom_lineage_range(self):
        """generate_random_fractalstat_address should respect lineage range."""
        from fractalstat.fractalstat_rag_bridge import (
            generate_random_fractalstat_address,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        addr = generate_random_fractalstat_address(realm, lineage_range=(5, 15))

        assert 5 <= addr.lineage <= 15

    def test_generate_synthetic_rag_documents(self):
        """generate_synthetic_rag_documents should create valid documents."""
        from fractalstat.fractalstat_rag_bridge import (
            generate_synthetic_rag_documents,
            Realm,
        )

        realm = Realm(type="game", label="Main")
        base_texts = ["text one", "text two"]

        def mock_embedding_fn(text):
            return [0.1, 0.2, 0.3]

        docs = generate_synthetic_rag_documents(
            base_texts=base_texts,
            realm=realm,
            scale=3,
            embedding_fn=mock_embedding_fn,
        )

        assert isinstance(docs, list)
        assert len(docs) >= len(base_texts)
        assert all(hasattr(doc, "id") for doc in docs)
        assert all(hasattr(doc, "text") for doc in docs)
        assert all(hasattr(doc, "embedding") for doc in docs)
        assert all(hasattr(doc, "fractalstat") for doc in docs)

    def test_compare_retrieval_results(self):
        """compare_retrieval_results should compare two result sets."""
        from fractalstat.fractalstat_rag_bridge import compare_retrieval_results

        semantic_results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        hybrid_results = [("doc1", 0.95), ("doc2", 0.65), ("doc3", 0.5)]

        comparison = compare_retrieval_results(semantic_results, hybrid_results, k=10)

        assert isinstance(comparison, dict)
        assert "overlap_count" in comparison
        assert "overlap_pct" in comparison
        assert "hybrid_avg_score" in comparison
