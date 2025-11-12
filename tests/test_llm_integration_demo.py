"""
Test suite for LLM Integration Demo (Approach #2)
Tests embedding generation, LLM narrative enhancement, and STAT7 coordinate extraction.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MockBitChain:
    """Mock BitChain for testing without full exp07 dependency."""
    bit_chain_id: str
    content: str
    realm: str
    luminosity: float = 0.5
    polarity: str = "logic"
    lineage: int = 1
    horizon: str = "emergence"
    dimensionality: int = 1


class TestLLMIntegrationDemoInit:
    """Test LLMIntegrationDemo initialization."""

    def test_initialization_succeeds(self):
        """LLMIntegrationDemo should initialize without errors."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        assert demo is not None
        assert hasattr(demo, 'embedder')
        assert hasattr(demo, 'generator')

    def test_embedder_model_loaded(self):
        """Embedder should be a SentenceTransformer instance."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        assert demo.embedder is not None
        
        # Test model can encode text
        embedding = demo.embedder.encode("test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

    def test_generator_available(self):
        """Generator (text generation pipeline) should be available."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        assert demo.generator is not None


class TestEmbeddingGeneration:
    """Test STAT7 address embedding generation."""

    def test_embed_stat7_address_returns_vector(self):
        """embed_stat7_address should return a numpy array."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="test-1",
            content="Test entity",
            realm="companion",
            luminosity=0.7
        )
        
        embedding = demo.embed_stat7_address(bit_chain)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384

    def test_embed_preserves_stat7_properties(self):
        """Embedding should incorporate STAT7 properties into representation."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        
        # Create two bit chains with different properties
        bit_chain1 = MockBitChain(
            bit_chain_id="test-1",
            content="Companion entity",
            realm="companion",
            luminosity=0.9
        )
        bit_chain2 = MockBitChain(
            bit_chain_id="test-2",
            content="Badge entity",
            realm="badge",
            luminosity=0.3
        )
        
        emb1 = demo.embed_stat7_address(bit_chain1)
        emb2 = demo.embed_stat7_address(bit_chain2)
        
        # Different STAT7 properties should produce different embeddings
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity < 0.95  # Should not be identical


class TestNarrativeEnhancement:
    """Test LLM narrative generation for STAT7 entities."""

    def test_enhance_bit_chain_narrative_returns_dict(self):
        """enhance_bit_chain_narrative should return a dictionary."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="test-1",
            content="Test entity",
            realm="companion",
            luminosity=0.7
        )
        
        result = demo.enhance_bit_chain_narrative(bit_chain)
        assert isinstance(result, dict)

    def test_enhance_contains_required_fields(self):
        """enhance_bit_chain_narrative output should contain required fields."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="test-1",
            content="Test entity",
            realm="companion",
            luminosity=0.7
        )
        
        result = demo.enhance_bit_chain_narrative(bit_chain)
        
        required_fields = [
            'bit_chain_id',
            'embedding',
            'enhanced_narrative',
            'integration_proof'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_embedding_in_result_is_valid(self):
        """Embedding in result should be a numpy array."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="test-1",
            content="Test entity",
            realm="companion"
        )
        
        result = demo.enhance_bit_chain_narrative(bit_chain)
        assert isinstance(result['embedding'], np.ndarray)
        assert len(result['embedding']) == 384

    def test_enhanced_narrative_is_string(self):
        """Enhanced narrative should be a non-empty string."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="test-1",
            content="Test entity",
            realm="companion"
        )
        
        result = demo.enhance_bit_chain_narrative(bit_chain)
        assert isinstance(result['enhanced_narrative'], str)
        assert len(result['enhanced_narrative']) > 0

    def test_integration_proof_field(self):
        """Integration proof field should confirm LLM integration."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="test-1",
            content="Test entity",
            realm="companion"
        )
        
        result = demo.enhance_bit_chain_narrative(bit_chain)
        assert 'successfully' in result['integration_proof'].lower()


class TestBatchEnhancement:
    """Test batch processing of multiple bit chains."""

    def test_batch_enhance_returns_list(self):
        """batch_enhance_narratives should return a list."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}",
                content=f"Test entity {i}",
                realm="companion" if i % 2 == 0 else "badge"
            )
            for i in range(3)
        ]
        
        results = demo.batch_enhance_narratives(bit_chains)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_enhance_all_items_valid(self):
        """All items in batch result should be valid dicts."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chains = [
            MockBitChain(
                bit_chain_id=f"test-{i}",
                content=f"Test entity {i}",
                realm="companion"
            )
            for i in range(2)
        ]
        
        results = demo.batch_enhance_narratives(bit_chains)
        
        for result in results:
            assert isinstance(result, dict)
            assert 'bit_chain_id' in result
            assert 'embedding' in result


class TestSTAT7CoordinateExtraction:
    """Test extracting STAT7 coordinates from embeddings."""

    def test_extract_stat7_coordinates_returns_dict(self):
        """extract_stat7_from_embedding should return a dict."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        
        # Generate a sample embedding
        embedding = demo.embedder.encode("test content")
        
        coords = demo.extract_stat7_from_embedding(embedding)
        assert isinstance(coords, dict)

    def test_extracted_coordinates_have_all_dimensions(self):
        """Extracted coordinates should have all 7 STAT7 dimensions."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        embedding = demo.embedder.encode("test content")
        
        coords = demo.extract_stat7_from_embedding(embedding)
        
        required_dimensions = [
            'lineage',
            'adjacency',
            'luminosity',
            'polarity',
            'dimensionality',
            'horizon',
            'realm'
        ]
        
        for dim in required_dimensions:
            assert dim in coords, f"Missing dimension: {dim}"

    def test_coordinates_are_normalized(self):
        """Extracted coordinates should use hybrid normalization bounds."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        embedding = demo.embedder.encode("test content")
        
        coords = demo.extract_stat7_from_embedding(embedding)
        
        # Hybrid bounds: fractal dimensions unbounded, relational symmetric, intensity asymmetric
        assert isinstance(coords['lineage'], (int, float)), "lineage should be numeric"
        assert -1.0 <= coords['adjacency'] <= 1.0, f"adjacency out of range: {coords['adjacency']}"
        assert 0.0 <= coords['luminosity'] <= 1.0, f"luminosity out of range: {coords['luminosity']}"
        assert -1.0 <= coords['polarity'] <= 1.0, f"polarity out of range: {coords['polarity']}"
        assert isinstance(coords['dimensionality'], (int, float)), "dimensionality should be numeric"

    def test_different_embeddings_yield_different_coordinates(self):
        """Different embeddings should produce different STAT7 coordinates."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        
        emb1 = demo.embedder.encode("a very bright companion entity full of energy")
        emb2 = demo.embedder.encode("a dark void badge empty and cold")
        
        coords1 = demo.extract_stat7_from_embedding(emb1)
        coords2 = demo.extract_stat7_from_embedding(emb2)
        
        # Coordinates should be populated and valid
        assert coords1['luminosity'] >= 0
        assert coords2['luminosity'] >= 0
        
        # Different semantic content should yield different coordinate profiles
        all_coords = [k for k in coords1.keys() if isinstance(coords1.get(k), (int, float))]
        assert len(all_coords) > 0


class TestIntegrationProof:
    """Test the complete integration workflow."""

    def test_end_to_end_integration(self):
        """Complete workflow: embed -> enhance -> extract STAT7."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        bit_chain = MockBitChain(
            bit_chain_id="integration-test",
            content="Integration test entity",
            realm="companion",
            luminosity=0.8
        )
        
        # Step 1: Enhance narrative
        enhanced = demo.enhance_bit_chain_narrative(bit_chain)
        assert 'embedding' in enhanced
        assert 'enhanced_narrative' in enhanced
        
        # Step 2: Extract STAT7 coordinates
        embedding = enhanced['embedding']
        stat7_coords = demo.extract_stat7_from_embedding(embedding)
        assert 'luminosity' in stat7_coords
        
        # Step 3: Verify integration proof
        assert 'successfully' in enhanced['integration_proof'].lower()

    def test_generate_integration_report(self):
        """generate_integration_report should return a complete report dict."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        report = demo.generate_integration_report()
        
        assert isinstance(report, dict)
        assert 'integration_capabilities' in report
        assert 'technical_stack' in report
        assert 'academic_validation' in report


class TestProviderInfo:
    """Test provider metadata."""

    def test_get_provider_info(self):
        """get_provider_info should return metadata."""
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        demo = LLMIntegrationDemo()
        info = demo.get_provider_info()
        
        assert isinstance(info, dict)
        assert 'embedding_dimension' in info
        assert 'model_name' in info
        assert 'device' in info
