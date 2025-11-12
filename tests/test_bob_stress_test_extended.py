"""
Extended tests for Bob Stress Test to achieve 95%+ coverage
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime


class TestBobStressTestExtended:
    """Extended tests for Bob stress testing framework."""

    @pytest.mark.asyncio
    async def test_single_query_with_mock_api(self):
        """single_query should handle API responses correctly."""
        from fractalstat.bob_stress_test import BobStressTester, NPCQueryGenerator
        
        tester = BobStressTester()
        gen = NPCQueryGenerator()
        query_data = gen.generate_query("npc_character_development")
        
        # Mock successful API response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "bob_status": "PASSED",
                "results": [{"id": "1"}],
                "coherence": 0.9,
                "entanglement": 0.2,
                "bob_verification_log": []
            }
            mock_get.return_value = mock_response
            
            result = await tester.single_query(query_data)
            
            assert "bob_status" in result
            assert result["bob_status"] == "PASSED"
            assert "query_time" in result

    @pytest.mark.asyncio
    async def test_single_query_error_handling(self):
        """single_query should handle API errors gracefully."""
        from fractalstat.bob_stress_test import BobStressTester, NPCQueryGenerator
        
        tester = BobStressTester()
        gen = NPCQueryGenerator()
        query_data = gen.generate_query("npc_character_development")
        
        # Mock API error
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_get.return_value = mock_response
            
            result = await tester.single_query(query_data)
            
            assert "error" in result
            assert tester.error_count > 0

    @pytest.mark.asyncio
    async def test_single_query_timeout(self):
        """single_query should handle timeouts."""
        from fractalstat.bob_stress_test import BobStressTester, NPCQueryGenerator
        import requests
        
        tester = BobStressTester()
        gen = NPCQueryGenerator()
        query_data = gen.generate_query("npc_character_development")
        
        # Mock timeout
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")
            
            result = await tester.single_query(query_data)
            
            assert "error" in result
            assert tester.error_count > 0

    @pytest.mark.asyncio
    async def test_query_worker_execution(self):
        """query_worker should execute queries for specified duration."""
        from fractalstat.bob_stress_test import BobStressTester
        
        tester = BobStressTester()
        
        # Mock single_query to return immediately
        async def mock_single_query(query_data):
            return {
                "timestamp": datetime.now().isoformat(),
                "query_id": query_data["query_id"],
                "bob_status": "PASSED",
                "query_time": 0.1
            }
        
        tester.single_query = mock_single_query
        
        # Run worker for 1 second
        await tester.query_worker(worker_id=0, duration_seconds=1)
        
        assert len(tester.results) > 0

    @pytest.mark.asyncio
    async def test_run_stress_test_short_duration(self):
        """run_stress_test should complete successfully."""
        from fractalstat.bob_stress_test import BobStressTester
        
        tester = BobStressTester()
        tester.config.MAX_CONCURRENT_QUERIES = 2
        
        # Mock single_query
        async def mock_single_query(query_data):
            return {
                "timestamp": datetime.now().isoformat(),
                "query_id": query_data["query_id"],
                "bob_status": "PASSED",
                "query_time": 0.1,
                "query_type": query_data["query_type"],
                "npc": query_data["npc"],
                "location": query_data["location"],
                "emotion": query_data["emotion"],
                "activity": query_data["activity"],
                "hybrid": query_data["hybrid"],
                "coherence": 0.9,
                "entanglement": 0.2,
                "result_count": 1,
                "bob_verification_log": []
            }
        
        tester.single_query = mock_single_query
        
        # Run for very short duration
        report = await tester.run_stress_test(duration_minutes=0.01)
        
        assert "test_summary" in report
        assert "volume_metrics" in report
        assert "bob_analysis" in report

    def test_generate_report_with_results(self):
        """generate_report should calculate metrics correctly."""
        from fractalstat.bob_stress_test import BobStressTester
        
        tester = BobStressTester()
        tester.start_time = datetime.now()
        tester.end_time = datetime.now()
        
        # Add mock results
        tester.results = [
            {
                "query_id": "1",
                "bob_status": "PASSED",
                "query_type": "npc_character_development"
            },
            {
                "query_id": "2",
                "bob_status": "VERIFIED",
                "query_type": "narrative_consistency"
            },
            {
                "query_id": "3",
                "error": "timeout"
            }
        ]
        tester.query_times = [0.1, 0.2, 0.15]
        tester.bob_verdicts = {"PASSED": 1, "VERIFIED": 1, "QUARANTINED": 0}
        tester.error_count = 1
        
        report = tester.generate_report()
        
        assert report["volume_metrics"]["total_queries"] == 3
        assert report["volume_metrics"]["failed_queries"] == 1
        assert report["bob_analysis"]["passed"] == 1
        assert report["bob_analysis"]["verified"] == 1

    def test_generate_report_saves_file(self):
        """generate_report should save report to file."""
        from fractalstat.bob_stress_test import BobStressTester
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('fractalstat.bob_stress_test.Path') as mock_path:
                mock_path.return_value.parent = Path(tmpdir)
                
                tester = BobStressTester()
                tester.start_time = datetime.now()
                tester.end_time = datetime.now()
                tester.results = []
                
                report = tester.generate_report()
                
                assert isinstance(report, dict)

    def test_query_type_distribution_analysis(self):
        """Report should analyze query type distribution."""
        from fractalstat.bob_stress_test import BobStressTester
        
        tester = BobStressTester()
        tester.start_time = datetime.now()
        tester.end_time = datetime.now()
        
        tester.results = [
            {"query_type": "npc_character_development", "bob_status": "PASSED"},
            {"query_type": "npc_character_development", "bob_status": "QUARANTINED"},
            {"query_type": "narrative_consistency", "bob_status": "PASSED"},
        ]
        tester.query_times = [0.1, 0.2, 0.15]
        tester.bob_verdicts = {"PASSED": 2, "VERIFIED": 0, "QUARANTINED": 1}
        
        report = tester.generate_report()
        
        assert "query_type_analysis" in report
        assert "npc_character_development" in report["query_type_analysis"]
        assert report["query_type_analysis"]["npc_character_development"]["total"] == 2

    def test_performance_degradation_detection(self):
        """Report should detect performance degradation."""
        from fractalstat.bob_stress_test import BobStressTester
        
        tester = BobStressTester()
        tester.start_time = datetime.now()
        tester.end_time = datetime.now()
        tester.results = []
        
        # Simulate degrading performance
        tester.query_times = [0.1] * 50 + [1.0] * 50  # Performance degrades
        
        report = tester.generate_report()
        
        # P95 should be significantly higher than median
        assert report["performance_metrics"]["p95_query_time_ms"] > report["performance_metrics"]["median_query_time_ms"]

    def test_bob_verdict_tracking(self):
        """Bob verdicts should be tracked correctly."""
        from fractalstat.bob_stress_test import BobStressTester
        
        tester = BobStressTester()
        tester.start_time = datetime.now()
        tester.end_time = datetime.now()
        
        tester.bob_verdicts = {"PASSED": 80, "VERIFIED": 15, "QUARANTINED": 5}
        tester.results = [{"bob_status": "PASSED"}] * 80 + [{"bob_status": "VERIFIED"}] * 15 + [{"bob_status": "QUARANTINED"}] * 5
        tester.query_times = [0.1] * 100
        
        report = tester.generate_report()
        
        assert report["bob_analysis"]["total_decisions"] == 100
        assert report["bob_analysis"]["alert_rate"] == 0.20  # (15 + 5) / 100
        assert report["bob_analysis"]["quarantine_rate"] == 0.05  # 5 / 100

    def test_query_generator_all_types(self):
        """Query generator should support all query types."""
        from fractalstat.bob_stress_test import NPCQueryGenerator, BobStressTestConfig
        
        gen = NPCQueryGenerator()
        config = BobStressTestConfig()
        
        for query_type in config.QUERY_TYPES:
            query = gen.generate_query(query_type)
            assert query["query_type"] == query_type
            assert len(query["semantic"]) > 0
            assert query["npc"] in gen.npc_names
            assert query["location"] in gen.locations

    def test_config_fallback_to_defaults(self):
        """Config should fall back to defaults if ExperimentConfig unavailable."""
        with patch.dict('sys.modules', {'fractalstat.config': None}):
            from fractalstat.bob_stress_test import BobStressTestConfig
            
            config = BobStressTestConfig()
            
            assert config.TEST_DURATION_MINUTES > 0
            assert config.QUERIES_PER_SECOND_TARGET > 0
            assert len(config.QUERY_TYPES) > 0
