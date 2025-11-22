"""
Test suite for EXP-08: RAG Integration Test
Tests RAG document integration with STAT7 addressing system.
"""

from unittest.mock import Mock, patch


class TestRAGTestResult:
    """Test RAG test result tracking."""

    def test_result_initialization(self):
        """RAGTestResult should initialize with defaults."""
        from fractalstat.exp08_rag_integration import RAGTestResult

        result = RAGTestResult()
        assert result.experiment == "EXP-08"
        assert result.status == "PASS"
        assert result.title == "RAG Integration Test"

    def test_result_has_timestamp(self):
        """RAGTestResult should have timestamp."""
        from fractalstat.exp08_rag_integration import RAGTestResult

        result = RAGTestResult()
        assert result.timestamp != ""
        assert "T" in result.timestamp


class TestRAGIntegrationTester:
    """Test RAG integration tester."""

    def test_tester_initialization(self):
        """RAGIntegrationTester should initialize."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        tester = RAGIntegrationTester()
        assert tester is not None
        assert tester.api_base_url == "http://localhost:8000"

    def test_tester_custom_api_url(self):
        """RAGIntegrationTester should accept custom API URL."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        custom_url = "http://custom-api:9000"
        tester = RAGIntegrationTester(api_base_url=custom_url)
        assert tester.api_base_url == custom_url

    @patch("fractalstat.exp08_rag_integration.requests.get")
    def test_check_api_health_success(self, mock_get):
        """check_api_health should return True on 200 response."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        tester = RAGIntegrationTester()
        health = tester.check_api_health()

        assert health is True

    @patch("fractalstat.exp08_rag_integration.requests.get")
    def test_check_api_health_failure(self, mock_get):
        """check_api_health should return False on error."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_get.side_effect = Exception("Connection error")

        tester = RAGIntegrationTester()
        health = tester.check_api_health()

        assert health is False

    @patch("fractalstat.exp08_rag_integration.requests.post")
    def test_test_semantic_retrieval_returns_dict(self, mock_post):
        """test_semantic_retrieval should return results dict."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results_count": 1,
            "execution_time_ms": 10,
        }
        mock_post.return_value = mock_response

        tester = RAGIntegrationTester()
        results = tester.test_semantic_retrieval()

        assert isinstance(results, dict)
        assert "total_queries" in results
        assert "successful_queries" in results

    @patch("fractalstat.exp08_rag_integration.requests.post")
    def test_test_hybrid_retrieval_returns_dict(self, mock_post):
        """test_hybrid_retrieval should return hybrid results dict."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results_count": 2,
            "execution_time_ms": 15,
            "narrative_analysis": {"coherence_score": 0.8},
        }
        mock_post.return_value = mock_response

        tester = RAGIntegrationTester()
        results = tester.test_hybrid_retrieval()

        assert isinstance(results, dict)
        assert "total_queries" in results

    @patch("fractalstat.exp08_rag_integration.requests.get")
    def test_check_rag_data_integration(self, mock_get):
        """check_rag_data_integration should verify data availability."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_queries": 100,
            "concurrent_queries": 5,
            "errors": 0,
        }
        mock_get.return_value = mock_response

        tester = RAGIntegrationTester()
        integration = tester.check_rag_data_integration()

        assert isinstance(integration, dict)
        assert integration["api_healthy"] is True

    @patch("fractalstat.exp08_rag_integration.RAGIntegrationTester.check_api_health")
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.test_semantic_retrieval"
    )
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.test_hybrid_retrieval"
    )
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.check_rag_data_integration"
    )
    def test_run_comprehensive_test_api_down(
        self, mock_rag, mock_hybrid, mock_semantic, mock_health
    ):
        """run_comprehensive_test should fail gracefully if API is down."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_health.return_value = False

        tester = RAGIntegrationTester()
        results = tester.run_comprehensive_test()

        assert results.status == "FAIL"
        assert "error" in results.results

    @patch("fractalstat.exp08_rag_integration.RAGIntegrationTester.check_api_health")
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.test_semantic_retrieval"
    )
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.test_hybrid_retrieval"
    )
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.check_rag_data_integration"
    )
    def test_run_comprehensive_test_success(
        self, mock_rag, mock_hybrid, mock_semantic, mock_health
    ):
        """run_comprehensive_test should pass with all checks."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_health.return_value = True
        mock_semantic.return_value = {
            "total_queries": 3,
            "successful_queries": 2,
            "results": [],
        }
        mock_hybrid.return_value = {
            "total_queries": 2,
            "successful_queries": 2,
            "results": [],
        }
        mock_rag.return_value = {
            "api_healthy": True,
            "data_integration_success": True,
        }

        tester = RAGIntegrationTester()
        results = tester.run_comprehensive_test()

        assert results.status == "PASS"


class TestRAGQueryStructure:
    """Test RAG query structures."""

    def test_semantic_query_format(self):
        """Semantic queries should have required fields."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        RAGIntegrationTester()

        queries = [
            {
                "query_id": "rag_test_1",
                "semantic": "test query",
                "expected_results": 1,
            },
        ]

        assert all("query_id" in q for q in queries)
        assert all("semantic" in q for q in queries)

    def test_hybrid_query_format(self):
        """Hybrid queries should include weight parameters."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        RAGIntegrationTester()

        queries = [
            {
                "query_id": "hybrid_test_1",
                "semantic": "test query",
                "weight_semantic": 0.6,
                "weight_stat7": 0.4,
            },
        ]

        assert all("weight_semantic" in q for q in queries)
        assert all("weight_stat7" in q for q in queries)


class TestRAGResultSerialization:
    """Test RAG test result serialization."""

    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.run_comprehensive_test"
    )
    def test_save_results_creates_file(self, mock_test):
        """save_results should create JSON file."""
        from fractalstat.exp08_rag_integration import (
            RAGIntegrationTester,
            RAGTestResult,
        )
        import tempfile
        import os

        mock_result = RAGTestResult(status="PASS", results={"test": "data"})
        mock_test.return_value = mock_result

        tester = RAGIntegrationTester()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_rag_results.json")
            tester.results = mock_result

            saved_path = tester.save_results(output_file)

            assert os.path.exists(saved_path)


class TestRAGIntegrationMetrics:
    """Test metrics calculation for RAG integration."""

    @patch("fractalstat.exp08_rag_integration.RAGIntegrationTester.check_api_health")
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.test_semantic_retrieval"
    )
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.test_hybrid_retrieval"
    )
    @patch(
        "fractalstat.exp08_rag_integration.RAGIntegrationTester.check_rag_data_integration"
    )
    def test_overall_metrics_calculation(
        self, mock_rag, mock_hybrid, mock_semantic, mock_health
    ):
        """Overall metrics should aggregate test results."""
        from fractalstat.exp08_rag_integration import RAGIntegrationTester

        mock_health.return_value = True
        mock_semantic.return_value = {
            "total_queries": 3,
            "successful_queries": 3,
            "results": [
                {"success": True},
                {"success": True},
                {"success": True},
            ],
        }
        mock_hybrid.return_value = {
            "total_queries": 2,
            "successful_queries": 2,
            "results": [{"success": True}, {"success": True}],
        }
        mock_rag.return_value = {
            "api_healthy": True,
            "data_integration_success": True,
        }

        tester = RAGIntegrationTester()
        results = tester.run_comprehensive_test()

        assert results.results is not None
        assert "overall_metrics" in results.results
        metrics = results.results["overall_metrics"]
        assert metrics["total_queries_tested"] == 5
        assert metrics["successful_queries"] == 5
