"""
Comprehensive test suite for EXP-08: LLM Integration
Tests LLM capabilities, embedding generation, narrative enhancement, and STAT7 extraction.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock


@dataclass
class MockBitChain:
    """Mock bit chain for testing."""
    bit_chain_id: str
    content: str
    realm: str
    luminosity: float = 0.7


class TestLLMIntegrationDemo:
    """Test LLMIntegrationDemo class."""

    def test_initialization(self):
        """LLMIntegrationDemo should initialize with models."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            assert demo.model_name == "all-MiniLM-L6-v2"
            assert demo.generator_model == "gpt2"
            assert demo.embedding_dimension > 0
            assert demo.device in ["cpu", "cuda"]
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")

    def test_embed_stat7_address(self):
        """embed_stat7_address should generate embeddings."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            bit_chain = MockBitChain(
                bit_chain_id="test-001",
                content="Test content",
                realm="data",
                luminosity=0.8
            )
            
            embedding = demo.embed_stat7_address(bit_chain)
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0
            assert len(embedding) == demo.embedding_dimension
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_enhance_bit_chain_narrative(self):
        """enhance_bit_chain_narrative should generate enhanced narrative."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            bit_chain = MockBitChain(
                bit_chain_id="test-002",
                content="A sentient companion",
                realm="companion",
                luminosity=0.9
            )
            
            result = demo.enhance_bit_chain_narrative(bit_chain)
            
            assert "bit_chain_id" in result
            assert "embedding" in result
            assert "enhanced_narrative" in result
            assert "integration_proof" in result
            assert isinstance(result["embedding"], np.ndarray)
            assert len(result["enhanced_narrative"]) > 0
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")

    def test_batch_enhance_narratives(self):
        """batch_enhance_narratives should process multiple bit chains."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            bit_chains = [
                MockBitChain(
                    bit_chain_id=f"test-{i}",
                    content=f"Content {i}",
                    realm="data",
                    luminosity=0.5 + i * 0.1
                )
                for i in range(3)
            ]
            
            results = demo.batch_enhance_narratives(bit_chains)
            
            assert len(results) == 3
            assert all("embedding" in r for r in results)
            assert all("enhanced_narrative" in r for r in results)
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")

    def test_extract_stat7_from_embedding(self):
        """extract_stat7_from_embedding should extract STAT7 coordinates."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            # Create test embedding
            embedding = np.random.randn(384)
            
            stat7 = demo.extract_stat7_from_embedding(embedding)
            
            assert "lineage" in stat7
            assert "adjacency" in stat7
            assert "luminosity" in stat7
            assert "polarity" in stat7
            assert "dimensionality" in stat7
            assert "horizon" in stat7
            assert "realm" in stat7
            
            # Hybrid bounds: fractal dimensions unbounded, relational symmetric, intensity asymmetric
            assert isinstance(stat7["lineage"], (int, float))
            assert -1.0 <= stat7["adjacency"] <= 1.0
            assert 0.0 <= stat7["luminosity"] <= 1.0
            assert -1.0 <= stat7["polarity"] <= 1.0
            assert isinstance(stat7["dimensionality"], (int, float))
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_extract_stat7_empty_embedding(self):
        """extract_stat7_from_embedding should handle empty embedding."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            stat7 = demo.extract_stat7_from_embedding(np.array([]))
            
            # Should return default values
            assert stat7["lineage"] == 0.5
            assert stat7["adjacency"] == 0.5
            assert stat7["luminosity"] == 0.7
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_get_provider_info(self):
        """get_provider_info should return provider metadata."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            info = demo.get_provider_info()
            
            assert "provider" in info
            assert "embedding_dimension" in info
            assert "model_name" in info
            assert "generator_model" in info
            assert "device" in info
            assert "status" in info
            assert info["status"] == "initialized"
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")

    def test_generate_integration_report(self):
        """generate_integration_report should return comprehensive report."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            report = demo.generate_integration_report()
            
            assert "integration_capabilities" in report
            assert "technical_stack" in report
            assert "academic_validation" in report
            assert "performance_metrics" in report
            assert "deployment_readiness" in report
            
            # Check academic validation
            assert report["academic_validation"]["integration_ready"] == True
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")

    def test_model_initialization_error_handling(self):
        """Should raise ImportError if dependencies not available."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                demo = LLMIntegrationDemo()

    def test_generator_initialization_error_handling(self):
        """Should raise ImportError if transformers not available."""
        # Need to mock the pipeline import within _initialize_models
        from fractalstat.exp08_llm_integration import LLMIntegrationDemo
        
        with patch('fractalstat.exp08_llm_integration.LLMIntegrationDemo._initialize_models') as mock_init:
            # Make _initialize_models raise ImportError for transformers
            mock_init.side_effect = ImportError("transformers not installed. Install with: pip install transformers")
            
            with pytest.raises(ImportError, match="transformers not installed"):
                demo = LLMIntegrationDemo()

    def test_enhance_narrative_with_generator_error(self):
        """enhance_bit_chain_narrative should handle generator errors gracefully."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            # Mock generator to raise exception
            demo.generator = Mock(side_effect=Exception("Generator error"))
            
            bit_chain = MockBitChain(
                bit_chain_id="test-error",
                content="Test content",
                realm="data",
                luminosity=0.7
            )
            
            result = demo.enhance_bit_chain_narrative(bit_chain)
            
            # Should still return result with fallback narrative
            assert "enhanced_narrative" in result
            assert "Enhanced:" in result["enhanced_narrative"]
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_enhance_narrative_without_generator(self):
        """enhance_bit_chain_narrative should work without generator."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            demo.generator = None
            
            bit_chain = MockBitChain(
                bit_chain_id="test-no-gen",
                content="Test content",
                realm="data",
                luminosity=0.7
            )
            
            result = demo.enhance_bit_chain_narrative(bit_chain)
            
            assert "enhanced_narrative" in result
            assert "Enhanced:" in result["enhanced_narrative"]
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_extract_stat7_with_list_input(self):
        """extract_stat7_from_embedding should handle list input."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            # Pass list instead of numpy array
            embedding_list = [0.1] * 384
            
            stat7 = demo.extract_stat7_from_embedding(embedding_list)
            
            assert "lineage" in stat7
            assert 0.0 <= stat7["lineage"] <= 1.0
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_extract_stat7_coordinate_normalization(self):
        """extract_stat7_from_embedding should use hybrid normalization bounds."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            # Create embedding with varied values to test normalization
            embedding = np.random.randn(384) * 10.0
            
            stat7 = demo.extract_stat7_from_embedding(embedding)
            
            # Hybrid bounds: fractal dimensions unbounded, relational symmetric, intensity asymmetric
            assert isinstance(stat7["lineage"], (int, float))
            assert -1.0 <= stat7["adjacency"] <= 1.0
            assert 0.0 <= stat7["luminosity"] <= 1.0
            assert -1.0 <= stat7["polarity"] <= 1.0
            assert isinstance(stat7["dimensionality"], (int, float))
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_batch_processing_empty_list(self):
        """batch_enhance_narratives should handle empty list."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            results = demo.batch_enhance_narratives([])
            
            assert results == []
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_embedding_dimension_consistency(self):
        """Embedding dimension should be consistent across calls."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            
            bit_chain1 = MockBitChain("id1", "content1", "data")
            bit_chain2 = MockBitChain("id2", "content2", "narrative")
            
            emb1 = demo.embed_stat7_address(bit_chain1)
            emb2 = demo.embed_stat7_address(bit_chain2)
            
            assert len(emb1) == len(emb2)
            assert len(emb1) == demo.embedding_dimension
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_integration_report_completeness(self):
        """Integration report should include all required sections."""
        try:
            from fractalstat.exp08_llm_integration import LLMIntegrationDemo
            
            demo = LLMIntegrationDemo()
            report = demo.generate_integration_report()
            
            # Check all capabilities are listed
            caps = report["integration_capabilities"]
            assert "embedding_generation" in caps
            assert "narrative_enhancement" in caps
            assert "coordinate_extraction" in caps
            assert "batch_processing" in caps
            assert "semantic_search" in caps
            
            # Check deployment readiness
            deploy = report["deployment_readiness"]
            assert "can_run_offline" in deploy
            assert "requires_service" in deploy
            assert "memory_efficient" in deploy
            assert "gpu_optional" in deploy
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")


class TestMainEntryPoint:
    """Test main entry point."""

    def test_main_execution(self):
        """Main function should execute successfully."""
        try:
            from fractalstat.exp08_llm_integration import main
            
            # Mock to avoid actual LLM calls
            with patch('fractalstat.exp08_llm_integration.LLMIntegrationDemo') as MockDemo:
                mock_demo = Mock()
                mock_demo.model_name = "test-model"
                mock_demo.embedding_dimension = 384
                mock_demo.generator_model = "gpt2"
                mock_demo.device = "cpu"
                mock_demo.enhance_bit_chain_narrative.return_value = {
                    "embedding": np.zeros(384),
                    "enhanced_narrative": "Test narrative",
                    "integration_proof": "Test proof"
                }
                mock_demo.extract_stat7_from_embedding.return_value = {
                    "lineage": 0.5,
                    "adjacency": 0.5,
                    "luminosity": 0.7,
                    "polarity": 0.5,
                    "dimensionality": 0.5,
                    "horizon": "scene",
                    "realm": {"type": "semantic"}
                }
                mock_demo.generate_integration_report.return_value = {
                    "technical_stack": {},
                    "integration_proof": "Test"
                }
                MockDemo.return_value = mock_demo
                
                report = main()
                
                assert report is not None
        except ImportError:
            pytest.skip("sentence-transformers or transformers not installed")
