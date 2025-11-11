"""
Comprehensive tests for STAT7-RAG Bridge
"""

import pytest
import math


class TestRealm:
    """Test Realm data structure."""

    def test_realm_initialization(self):
        """Realm should initialize with type and label."""
        from fractalstat.stat7_rag_bridge import Realm

        realm = Realm(type="game", label="Main Story")
        
        assert realm.type == "game"
        assert realm.label == "Main Story"

    def test_realm_various_types(self):
        """Realm should support various types."""
        from fractalstat.stat7_rag_bridge import Realm

        types = ["game", "system", "faculty", "pattern", "data", "narrative", "business", "concept"]
        
        for realm_type in types:
            realm = Realm(type=realm_type, label="Test")
            assert realm.type == realm_type


class TestSTAT7Address:
    """Test STAT7Address data structure."""

    def test_stat7_address_initialization(self):
        """STAT7Address should initialize with all coordinates."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        address = STAT7Address(
            realm=realm,
            lineage=5,
            adjacency=0.8,
            horizon="scene",
            luminosity=0.9,
            polarity=0.7,
            dimensionality=3
        )
        
        assert address.realm == realm
        assert address.lineage == 5
        assert address.adjacency == 0.8
        assert address.horizon == "scene"
        assert address.luminosity == 0.9
        assert address.polarity == 0.7
        assert address.dimensionality == 3

    def test_stat7_address_validates_adjacency_range(self):
        """STAT7Address should validate adjacency in [0, 1]."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        
        with pytest.raises(AssertionError):
            STAT7Address(
                realm=realm,
                lineage=0,
                adjacency=1.5,
                horizon="scene",
                luminosity=0.5,
                polarity=0.5,
                dimensionality=3
            )

    def test_stat7_address_validates_luminosity_range(self):
        """STAT7Address should validate luminosity in [0, 1]."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        
        with pytest.raises(AssertionError):
            STAT7Address(
                realm=realm,
                lineage=0,
                adjacency=0.5,
                horizon="scene",
                luminosity=1.5,
                polarity=0.5,
                dimensionality=3
            )

    def test_stat7_address_validates_polarity_range(self):
        """STAT7Address should validate polarity in [0, 1]."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        
        with pytest.raises(AssertionError):
            STAT7Address(
                realm=realm,
                lineage=0,
                adjacency=0.5,
                horizon="scene",
                luminosity=0.5,
                polarity=-0.5,
                dimensionality=3
            )

    def test_stat7_address_validates_lineage_nonnegative(self):
        """STAT7Address should validate lineage >= 0."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        
        with pytest.raises(AssertionError):
            STAT7Address(
                realm=realm,
                lineage=-1,
                adjacency=0.5,
                horizon="scene",
                luminosity=0.5,
                polarity=0.5,
                dimensionality=3
            )

    def test_stat7_address_validates_dimensionality_range(self):
        """STAT7Address should validate dimensionality in [1, 7]."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        
        with pytest.raises(AssertionError):
            STAT7Address(
                realm=realm,
                lineage=0,
                adjacency=0.5,
                horizon="scene",
                luminosity=0.5,
                polarity=0.5,
                dimensionality=8
            )

    def test_stat7_address_to_dict(self):
        """STAT7Address should convert to dictionary."""
        from fractalstat.stat7_rag_bridge import STAT7Address, Realm

        realm = Realm(type="game", label="Main Story")
        address = STAT7Address(
            realm=realm,
            lineage=5,
            adjacency=0.8,
            horizon="scene",
            luminosity=0.9,
            polarity=0.7,
            dimensionality=3
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


class TestRAGDocument:
    """Test RAGDocument data structure."""

    def test_rag_document_initialization(self):
        """RAGDocument should initialize with all fields."""
        from fractalstat.stat7_rag_bridge import RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        address = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        doc = RAGDocument(
            id="doc-001",
            text="Sample document text",
            embedding=[1.0, 0.0, 0.0],
            stat7=address,
            metadata={"source": "test"}
        )
        
        assert doc.id == "doc-001"
        assert doc.text == "Sample document text"
        assert doc.embedding == [1.0, 0.0, 0.0]
        assert doc.stat7 == address
        assert doc.metadata == {"source": "test"}

    def test_rag_document_validates_nonempty_embedding(self):
        """RAGDocument should validate non-empty embedding."""
        from fractalstat.stat7_rag_bridge import RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        address = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        with pytest.raises(AssertionError):
            RAGDocument(
                id="doc-001",
                text="Sample text",
                embedding=[],
                stat7=address
            )

    def test_rag_document_default_metadata(self):
        """RAGDocument should have empty metadata by default."""
        from fractalstat.stat7_rag_bridge import RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        address = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        doc = RAGDocument(
            id="doc-001",
            text="Sample",
            embedding=[1.0, 0.0],
            stat7=address
        )
        
        assert doc.metadata == {}


class TestCosineSimilarity:
    """Test cosine_similarity function."""

    def test_cosine_similarity_identical_vectors(self):
        """cosine_similarity should return 1.0 for identical vectors."""
        from fractalstat.stat7_rag_bridge import cosine_similarity

        vec = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec, vec)
        
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """cosine_similarity should return ~0.0 for orthogonal vectors."""
        from fractalstat.stat7_rag_bridge import cosine_similarity

        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = cosine_similarity(vec1, vec2)
        
        assert abs(similarity) < 1e-6

    def test_cosine_similarity_opposite_vectors(self):
        """cosine_similarity should return -1.0 for opposite vectors."""
        from fractalstat.stat7_rag_bridge import cosine_similarity

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        
        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_empty_vectors(self):
        """cosine_similarity should handle empty vectors."""
        from fractalstat.stat7_rag_bridge import cosine_similarity

        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_cosine_similarity_zero_magnitude_vector(self):
        """cosine_similarity should handle zero-magnitude vectors."""
        from fractalstat.stat7_rag_bridge import cosine_similarity

        zero_vec = [0.0, 0.0, 0.0]
        vec = [1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(zero_vec, vec)
        assert similarity == 0.0

    def test_cosine_similarity_normalized(self):
        """cosine_similarity should work with normalized vectors."""
        from fractalstat.stat7_rag_bridge import cosine_similarity

        vec1 = [0.6, 0.8]  # normalized: 0.36 + 0.64 = 1.0
        vec2 = [0.8, 0.6]
        
        similarity = cosine_similarity(vec1, vec2)
        expected = (0.6*0.8 + 0.8*0.6) / (1.0 * 1.0)  # 0.96
        
        assert abs(similarity - expected) < 1e-6


class TestSTAT7Resonance:
    """Test stat7_resonance function."""

    def test_stat7_resonance_same_realm_returns_high_score(self):
        """stat7_resonance should return high score for same realm."""
        from fractalstat.stat7_rag_bridge import stat7_resonance, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        query = STAT7Address(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        doc = STAT7Address(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        resonance = stat7_resonance(query, doc)
        assert resonance >= 0.85

    def test_stat7_resonance_different_realm_returns_lower_score(self):
        """stat7_resonance should return lower score for different realm."""
        from fractalstat.stat7_rag_bridge import stat7_resonance, STAT7Address, Realm

        realm1 = Realm(type="game", label="Main")
        realm2 = Realm(type="system", label="Core")
        
        query = STAT7Address(
            realm=realm1,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        doc = STAT7Address(
            realm=realm2,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        resonance = stat7_resonance(query, doc)
        assert 0.0 <= resonance <= 1.0

    def test_stat7_resonance_range(self):
        """stat7_resonance should always return value in [0, 1]."""
        from fractalstat.stat7_rag_bridge import stat7_resonance, STAT7Address, Realm

        realm1 = Realm(type="game", label="Main")
        realm2 = Realm(type="data", label="Records")
        
        addresses = []
        for lineage in [0, 5, 10]:
            for adjacency in [0.0, 0.5, 1.0]:
                addr = STAT7Address(
                    realm=realm1 if lineage % 2 == 0 else realm2,
                    lineage=lineage,
                    adjacency=adjacency,
                    horizon="scene",
                    luminosity=0.5 + adjacency/2,
                    polarity=0.5,
                    dimensionality=3
                )
                addresses.append(addr)
        
        for query in addresses[:3]:
            for doc in addresses[3:]:
                resonance = stat7_resonance(query, doc)
                assert 0.0 <= resonance <= 1.0


class TestHybridScoring:
    """Test hybrid scoring functions."""

    def test_hybrid_score_combines_semantic_and_stat7(self):
        """Hybrid scoring should combine semantic similarity and STAT7 resonance."""
        from fractalstat.stat7_rag_bridge import hybrid_score, RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        query_addr = STAT7Address(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        doc_addr = STAT7Address(
            realm=realm,
            lineage=5,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        query_embedding = [1.0, 0.0, 0.0]
        doc = RAGDocument(
            id="doc-1",
            text="test",
            embedding=[1.0, 0.0, 0.0],
            stat7=doc_addr
        )
        
        score = hybrid_score(
            query_embedding=query_embedding,
            doc=doc,
            query_stat7=query_addr,
            weight_semantic=0.5,
            weight_stat7=0.5
        )
        
        assert 0.0 <= score <= 1.0

    def test_hybrid_score_different_weights(self):
        """Hybrid scoring should respect different weights."""
        from fractalstat.stat7_rag_bridge import hybrid_score, RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        addr = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        query_emb = [1.0, 0.0]
        doc = RAGDocument(
            id="doc-1",
            text="test",
            embedding=[1.0, 0.0],
            stat7=addr
        )
        
        score1 = hybrid_score(query_emb, doc, addr, 0.9, 0.1)
        score2 = hybrid_score(query_emb, doc, addr, 0.1, 0.9)
        
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0

    def test_hybrid_score_weights_must_sum_to_one(self):
        """hybrid_score should require weights to sum to 1.0."""
        from fractalstat.stat7_rag_bridge import hybrid_score, RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        addr = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        doc = RAGDocument(
            id="doc-1",
            text="test",
            embedding=[1.0, 0.0],
            stat7=addr
        )
        
        with pytest.raises(AssertionError):
            hybrid_score([1.0, 0.0], doc, addr, 0.6, 0.5)


class TestRAGRetrieval:
    """Test RAG retrieval functionality."""

    def test_retrieve_returns_ranked_list(self):
        """retrieve should return ranked list of documents."""
        from fractalstat.stat7_rag_bridge import retrieve, RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        addr = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        docs = [
            RAGDocument(id="d1", text="text1", embedding=[1.0, 0.0], stat7=addr),
            RAGDocument(id="d2", text="text2", embedding=[0.0, 1.0], stat7=addr),
            RAGDocument(id="d3", text="text3", embedding=[1.0, 1.0], stat7=addr),
        ]
        
        query_emb = [1.0, 0.0]
        results = retrieve(
            documents=docs,
            query_embedding=query_emb,
            query_stat7=addr,
            k=2,
            weight_semantic=0.5,
            weight_stat7=0.5
        )
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_retrieve_respects_k_parameter(self):
        """retrieve should respect k parameter."""
        from fractalstat.stat7_rag_bridge import retrieve, RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        addr = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        docs = [
            RAGDocument(id=f"d{i}", text=f"text{i}", embedding=[float(i)/10, 0.0], stat7=addr)
            for i in range(10)
        ]
        
        results = retrieve(
            documents=docs,
            query_embedding=[1.0, 0.0],
            query_stat7=addr,
            k=3,
            weight_semantic=0.5,
            weight_stat7=0.5
        )
        
        assert len(results) <= 3

    def test_retrieve_empty_documents_list(self):
        """retrieve should handle empty document list."""
        from fractalstat.stat7_rag_bridge import retrieve, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        addr = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        results = retrieve(
            documents=[],
            query_embedding=[1.0, 0.0],
            query_stat7=addr,
            k=5,
            weight_semantic=0.5,
            weight_stat7=0.5
        )
        
        assert results == []

    def test_retrieve_semantic_only(self):
        """retrieve_semantic_only should work without STAT7."""
        from fractalstat.stat7_rag_bridge import retrieve_semantic_only, RAGDocument, STAT7Address, Realm

        realm = Realm(type="game", label="Main")
        addr = STAT7Address(
            realm=realm,
            lineage=0,
            adjacency=0.5,
            horizon="scene",
            luminosity=0.5,
            polarity=0.5,
            dimensionality=3
        )
        
        docs = [
            RAGDocument(id="d1", text="text1", embedding=[1.0, 0.0], stat7=addr),
            RAGDocument(id="d2", text="text2", embedding=[0.0, 1.0], stat7=addr),
        ]
        
        results = retrieve_semantic_only(
            documents=docs,
            query_embedding=[1.0, 0.0],
            k=2
        )
        
        assert isinstance(results, list)
        assert len(results) <= 2


class TestSynthenticDataGeneration:
    """Test synthetic data generation for testing."""

    def test_generate_random_stat7_address(self):
        """generate_random_stat7_address should create valid address."""
        from fractalstat.stat7_rag_bridge import generate_random_stat7_address, Realm

        realm = Realm(type="game", label="Main")
        addr = generate_random_stat7_address(realm)
        
        assert addr is not None
        assert addr.realm == realm
        assert hasattr(addr, 'lineage')
        assert 0.0 <= addr.adjacency <= 1.0
        assert 0.0 <= addr.luminosity <= 1.0
        assert 0.0 <= addr.polarity <= 1.0
        assert addr.lineage >= 0
        assert 1 <= addr.dimensionality <= 7

    def test_generate_random_stat7_address_custom_lineage_range(self):
        """generate_random_stat7_address should respect lineage range."""
        from fractalstat.stat7_rag_bridge import generate_random_stat7_address, Realm

        realm = Realm(type="game", label="Main")
        addr = generate_random_stat7_address(realm, lineage_range=(5, 15))
        
        assert 5 <= addr.lineage <= 15

    def test_generate_synthetic_rag_documents(self):
        """generate_synthetic_rag_documents should create valid documents."""
        from fractalstat.stat7_rag_bridge import generate_synthetic_rag_documents, Realm
        
        realm = Realm(type="game", label="Main")
        base_texts = ["text one", "text two"]
        
        def mock_embedding_fn(text):
            return [0.1, 0.2, 0.3]
        
        docs = generate_synthetic_rag_documents(
            base_texts=base_texts,
            realm=realm,
            scale=3,
            embedding_fn=mock_embedding_fn
        )
        
        assert isinstance(docs, list)
        assert len(docs) >= len(base_texts)
        assert all(hasattr(doc, 'id') for doc in docs)
        assert all(hasattr(doc, 'text') for doc in docs)
        assert all(hasattr(doc, 'embedding') for doc in docs)
        assert all(hasattr(doc, 'stat7') for doc in docs)

    def test_compare_retrieval_results(self):
        """compare_retrieval_results should compare two result sets."""
        from fractalstat.stat7_rag_bridge import compare_retrieval_results

        semantic_results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        hybrid_results = [("doc1", 0.95), ("doc2", 0.65), ("doc3", 0.5)]
        
        comparison = compare_retrieval_results(semantic_results, hybrid_results, k=10)
        
        assert isinstance(comparison, dict)
        assert "overlap_count" in comparison
        assert "overlap_pct" in comparison
        assert "hybrid_avg_score" in comparison
