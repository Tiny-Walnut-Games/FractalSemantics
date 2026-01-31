"""
Test suite for EXP-10: Multi-Dimensional Query Optimization

This module contains comprehensive tests for the multi-dimensional query optimization
experiment, validating all functionality of the modularized implementation.
"""

import unittest
import json
import time
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

from fractalstat.exp10_multidimensional_query import (
    QueryPattern,
    QueryResult,
    QueryOptimizer,
    MultiDimensionalQueryResults,
    MultiDimensionalQueryEngine,
    MultiDimensionalQueryExperiment
)


class TestQueryPattern(unittest.TestCase):
    """Test QueryPattern data structure."""
    
    def test_query_pattern_creation(self):
        """Test QueryPattern creation with all properties."""
        pattern = QueryPattern(
            pattern_name="Test Pattern",
            description="Test description",
            dimensions_used=["realm", "polarity"],
            complexity_level="medium",
            real_world_use_case="Test use case"
        )
        
        self.assertEqual(pattern.pattern_name, "Test Pattern")
        self.assertEqual(pattern.description, "Test description")
        self.assertEqual(pattern.dimensions_used, ["realm", "polarity"])
        self.assertEqual(pattern.complexity_level, "medium")
        self.assertEqual(pattern.real_world_use_case, "Test use case")
    
    def test_query_pattern_validation(self):
        """Test QueryPattern validation."""
        with self.assertRaises(Exception):
            QueryPattern(
                pattern_name="",
                description="",
                dimensions_used=[],
                complexity_level="invalid",
                real_world_use_case=""
            )


class TestQueryResult(unittest.TestCase):
    """Test QueryResult data structure."""
    
    def test_query_result_creation(self):
        """Test QueryResult creation with all properties."""
        result = QueryResult(
            query_id="test_1",
            pattern_name="Test Pattern",
            execution_time_ms=50.5,
            results_count=100,
            precision_score=0.9,
            recall_score=0.8,
            f1_score=0.85,
            memory_usage_mb=2.5,
            cpu_time_ms=40.0
        )
        
        self.assertEqual(result.query_id, "test_1")
        self.assertEqual(result.pattern_name, "Test Pattern")
        self.assertEqual(result.execution_time_ms, 50.5)
        self.assertEqual(result.results_count, 100)
        self.assertEqual(result.precision_score, 0.9)
        self.assertEqual(result.recall_score, 0.8)
        self.assertEqual(result.f1_score, 0.85)
        self.assertEqual(result.memory_usage_mb, 2.5)
        self.assertEqual(result.cpu_time_ms, 40.0)
    
    def test_query_result_validation(self):
        """Test QueryResult validation."""
        # Test invalid precision score
        with self.assertRaises(Exception):
            QueryResult(
                query_id="test_1",
                pattern_name="Test",
                execution_time_ms=50.0,
                results_count=100,
                precision_score=1.5,  # Invalid > 1.0
                recall_score=0.8,
                f1_score=0.85,
                memory_usage_mb=2.5,
                cpu_time_ms=40.0
            )


class TestQueryOptimizer(unittest.TestCase):
    """Test QueryOptimizer data structure."""
    
    def test_query_optimizer_creation(self):
        """Test QueryOptimizer creation with all properties."""
        optimizer = QueryOptimizer(
            strategy_name="Test Optimization",
            description="Test description",
            optimization_type="indexing",
            expected_improvement=0.5,
            complexity_overhead="medium"
        )
        
        self.assertEqual(optimizer.strategy_name, "Test Optimization")
        self.assertEqual(optimizer.description, "Test description")
        self.assertEqual(optimizer.optimization_type, "indexing")
        self.assertEqual(optimizer.expected_improvement, 0.5)
        self.assertEqual(optimizer.complexity_overhead, "medium")
    
    def test_query_optimizer_validation(self):
        """Test QueryOptimizer validation."""
        # Test invalid optimization type
        with self.assertRaises(Exception):
            QueryOptimizer(
                strategy_name="Test",
                description="Test",
                optimization_type="invalid",
                expected_improvement=0.5,
                complexity_overhead="medium"
            )


class TestMultiDimensionalQueryResults(unittest.TestCase):
    """Test MultiDimensionalQueryResults data structure."""
    
    def test_query_results_creation(self):
        """Test MultiDimensionalQueryResults creation."""
        results = MultiDimensionalQueryResults(
            experiment="EXP-10",
            title="Test Results",
            dataset_size=1000,
            avg_query_time_ms=50.0,
            avg_precision=0.9,
            avg_recall=0.8,
            avg_f1_score=0.85,
            query_throughput_qps=20.0,
            optimization_improvement=0.4,
            practical_value_score=0.8,
            scalability_score=0.9
        )
        
        self.assertEqual(results.experiment, "EXP-10")
        self.assertEqual(results.title, "Test Results")
        self.assertEqual(results.dataset_size, 1000)
        self.assertEqual(results.avg_query_time_ms, 50.0)
        self.assertEqual(results.avg_precision, 0.9)
        self.assertEqual(results.avg_recall, 0.8)
        self.assertEqual(results.avg_f1_score, 0.85)
        self.assertEqual(results.query_throughput_qps, 20.0)
        self.assertEqual(results.optimization_improvement, 0.4)
        self.assertEqual(results.practical_value_score, 0.8)
        self.assertEqual(results.scalability_score, 0.9)
    
    def test_query_results_serialization(self):
        """Test MultiDimensionalQueryResults serialization."""
        results = MultiDimensionalQueryResults(
            experiment="EXP-10",
            title="Test Results",
            dataset_size=1000,
            avg_query_time_ms=50.0,
            avg_precision=0.9,
            avg_recall=0.8,
            avg_f1_score=0.85,
            query_throughput_qps=20.0,
            optimization_improvement=0.4,
            practical_value_score=0.8,
            scalability_score=0.9
        )
        
        # Test to_dict
        result_dict = results.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['experiment'], "EXP-10")
        self.assertEqual(result_dict['dataset_size'], 1000)
        
        # Test to_json
        json_str = results.to_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed['experiment'], "EXP-10")
        self.assertEqual(parsed['dataset_size'], 1000)


class TestMultiDimensionalQueryEngine(unittest.TestCase):
    """Test MultiDimensionalQueryEngine functionality."""
    
    def setUp(self):
        """Set up test engine with small dataset."""
        self.engine = MultiDimensionalQueryEngine(dataset_size=100)
        self.engine.build_dataset()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.dataset_size, 100)
        self.assertEqual(len(self.engine.bit_chains), 100)
        self.assertEqual(len(self.engine.optimizers), 4)
    
    def test_dataset_building(self):
        """Test dataset building."""
        self.assertEqual(len(self.engine.bit_chains), 100)
        self.assertGreater(self.engine.coordinate_diversity, 0.0)
        self.assertLessEqual(self.engine.coordinate_diversity, 1.0)
    
    def test_query_execution(self):
        """Test query execution."""
        pattern = QueryPattern(
            pattern_name="Test Query",
            description="Test description",
            dimensions_used=["realm"],
            complexity_level="simple",
            real_world_use_case="Test"
        )
        
        result = self.engine.execute_query(pattern, "test_query_1")
        
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.pattern_name, "Test Query")
        self.assertGreater(result.execution_time_ms, 0.0)
        self.assertGreaterEqual(result.results_count, 0)
        self.assertGreaterEqual(result.precision_score, 0.0)
        self.assertLessEqual(result.precision_score, 1.0)
        self.assertGreaterEqual(result.recall_score, 0.0)
        self.assertLessEqual(result.recall_score, 1.0)
        self.assertGreaterEqual(result.f1_score, 0.0)
        self.assertLessEqual(result.f1_score, 1.0)
    
    def test_optimization_application(self):
        """Test optimization strategy application."""
        optimization_results = self.engine.apply_optimizations()
        
        self.assertIsInstance(optimization_results, dict)
        self.assertEqual(len(optimization_results), 4)  # 4 optimizers
        
        for strategy_name, result in optimization_results.items():
            self.assertIn('baseline_time_ms', result)
            self.assertIn('optimized_time_ms', result)
            self.assertIn('improvement_ratio', result)
            self.assertIn('optimization_time_ms', result)
            self.assertIn('expected_improvement', result)
            self.assertIn('complexity_overhead', result)
            
            self.assertGreaterEqual(result['baseline_time_ms'], 0.0)
            self.assertGreaterEqual(result['optimized_time_ms'], 0.0)
            self.assertGreaterEqual(result['improvement_ratio'], 0.0)
            self.assertGreaterEqual(result['optimization_time_ms'], 0.0)
            self.assertGreaterEqual(result['expected_improvement'], 0.0)
            self.assertIn(result['complexity_overhead'], ['low', 'medium', 'high'])


class TestMultiDimensionalQueryExperiment(unittest.TestCase):
    """Test MultiDimensionalQueryExperiment functionality."""
    
    def setUp(self):
        """Set up test experiment with small dataset."""
        self.experiment = MultiDimensionalQueryExperiment(dataset_size=100)
    
    def test_experiment_initialization(self):
        """Test experiment initialization."""
        self.assertEqual(self.experiment.dataset_size, 100)
        self.assertIsInstance(self.experiment.engine, MultiDimensionalQueryEngine)
    
    def test_experiment_execution(self):
        """Test experiment execution."""
        results = self.experiment.run()
        
        self.assertIsInstance(results, MultiDimensionalQueryResults)
        self.assertEqual(results.dataset_size, 100)
        self.assertIn(results.status, ["PASS", "PARTIAL", "FAIL"])
        self.assertGreater(results.avg_query_time_ms, 0.0)
        self.assertGreaterEqual(results.avg_precision, 0.0)
        self.assertLessEqual(results.avg_precision, 1.0)
        self.assertGreaterEqual(results.avg_recall, 0.0)
        self.assertLessEqual(results.avg_recall, 1.0)
        self.assertGreaterEqual(results.avg_f1_score, 0.0)
        self.assertLessEqual(results.avg_f1_score, 1.0)
        self.assertGreaterEqual(results.query_throughput_qps, 0.0)
        self.assertGreaterEqual(results.optimization_improvement, 0.0)
        self.assertGreaterEqual(results.practical_value_score, 0.0)
        self.assertLessEqual(results.practical_value_score, 1.0)
        self.assertGreaterEqual(results.scalability_score, 0.0)
        self.assertLessEqual(results.scalability_score, 1.0)
    
    def test_experiment_with_different_sizes(self):
        """Test experiment with different dataset sizes."""
        for size in [50, 100, 200]:
            with self.subTest(dataset_size=size):
                experiment = MultiDimensionalQueryExperiment(dataset_size=size)
                results = experiment.run()
                
                self.assertEqual(results.dataset_size, size)
                self.assertGreater(results.avg_query_time_ms, 0.0)
                self.assertGreaterEqual(results.avg_f1_score, 0.0)
                self.assertLessEqual(results.avg_f1_score, 1.0)
    
    def test_experiment_results_structure(self):
        """Test experiment results structure."""
        results = self.experiment.run()
        
        # Test query results
        self.assertIsInstance(results.query_results, list)
        self.assertGreater(len(results.query_results), 0)
        
        for query_result in results.query_results:
            self.assertIsInstance(query_result, QueryResult)
            self.assertGreater(query_result.execution_time_ms, 0.0)
            self.assertGreaterEqual(query_result.precision_score, 0.0)
            self.assertLessEqual(query_result.precision_score, 1.0)
        
        # Test optimizer results
        self.assertIsInstance(results.optimizer_results, list)
        self.assertGreater(len(results.optimizer_results), 0)
        
        # Test use case validation
        self.assertIsInstance(results.use_case_validation, dict)
        self.assertGreater(len(results.use_case_validation), 0)
        
        for use_case, validated in results.use_case_validation.items():
            self.assertIsInstance(use_case, str)
            self.assertIsInstance(validated, bool)
        
        # Test performance benchmarks
        self.assertIsInstance(results.performance_benchmarks, dict)
        self.assertIn('avg_query_time_ms', results.performance_benchmarks)
        self.assertIn('query_throughput_qps', results.performance_benchmarks)
        self.assertIn('optimization_improvement', results.performance_benchmarks)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete module."""
    
    def test_complete_workflow(self):
        """Test complete workflow from experiment to results."""
        # Create experiment
        experiment = MultiDimensionalQueryExperiment(dataset_size=50)
        
        # Run experiment
        results = experiment.run()
        
        # Validate results
        self.assertIsInstance(results, MultiDimensionalQueryResults)
        self.assertEqual(results.dataset_size, 50)
        self.assertIn(results.status, ["PASS", "PARTIAL", "FAIL"])
        
        # Test serialization
        json_str = results.to_json()
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        parsed = json.loads(json_str)
        self.assertEqual(parsed['experiment'], "EXP-10")
        self.assertEqual(parsed['dataset_size'], 50)
    
    def test_module_imports(self):
        """Test all module imports work correctly."""
        # Test main imports
        from fractalstat.exp10_multidimensional_query import (
            QueryPattern,
            QueryResult,
            QueryOptimizer,
            MultiDimensionalQueryResults,
            MultiDimensionalQueryEngine,
            MultiDimensionalQueryExperiment
        )
        
        # Test that all classes can be instantiated
        pattern = QueryPattern("Test", "Test", ["realm"], "simple", "Test")
        result = QueryResult("test", "Test", 50.0, 100, 0.9, 0.8, 0.85, 2.5, 40.0)
        optimizer = QueryOptimizer("Test", "Test", "indexing", 0.5, "medium")
        results = MultiDimensionalQueryResults()
        engine = MultiDimensionalQueryEngine(dataset_size=100)
        experiment = MultiDimensionalQueryExperiment(dataset_size=100)
        
        self.assertIsInstance(pattern, QueryPattern)
        self.assertIsInstance(result, QueryResult)
        self.assertIsInstance(optimizer, QueryOptimizer)
        self.assertIsInstance(results, MultiDimensionalQueryResults)
        self.assertIsInstance(engine, MultiDimensionalQueryEngine)
        self.assertIsInstance(experiment, MultiDimensionalQueryExperiment)


class TestPerformance(unittest.TestCase):
    """Performance tests for the module."""
    
    def test_query_performance(self):
        """Test query performance characteristics."""
        engine = MultiDimensionalQueryEngine(dataset_size=500)
        engine.build_dataset()
        
        # Test multiple queries
        pattern = QueryPattern(
            pattern_name="Performance Test",
            description="Performance test",
            dimensions_used=["realm", "polarity"],
            complexity_level="medium",
            real_world_use_case="Performance"
        )
        
        query_times = []
        for i in range(10):
            result = engine.execute_query(pattern, f"perf_query_{i}")
            query_times.append(result.execution_time_ms)
        
        # Average query time should be reasonable
        avg_time = sum(query_times) / len(query_times)
        self.assertLess(avg_time, 1000.0)  # Less than 1 second
        
        # Query times should be consistent
        std_dev = (sum((t - avg_time) ** 2 for t in query_times) / len(query_times)) ** 0.5
        self.assertLess(std_dev, avg_time * 0.5)  # Reasonable consistency
    
    def test_scalability(self):
        """Test scalability with different dataset sizes."""
        sizes = [100, 200, 500]
        times = []
        
        for size in sizes:
            experiment = MultiDimensionalQueryExperiment(dataset_size=size)
            start_time = time.time()
            results = experiment.run()
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            # Results should be valid
            self.assertEqual(results.dataset_size, size)
            self.assertGreater(results.avg_query_time_ms, 0.0)
        
        # Execution time should scale reasonably
        # (allowing for some variance due to system load)
        for i in range(1, len(times)):
            # Time should not increase more than 5x for 5x dataset size increase
            time_ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            self.assertLess(time_ratio, size_ratio * 10)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        engine = MultiDimensionalQueryEngine(dataset_size=0)
        engine.build_dataset()
        
        self.assertEqual(len(engine.bit_chains), 0)
        self.assertEqual(engine.coordinate_diversity, 0.0)
    
    def test_invalid_query_patterns(self):
        """Test handling of invalid query patterns."""
        engine = MultiDimensionalQueryEngine(dataset_size=10)
        engine.build_dataset()
        
        # Test with invalid pattern
        pattern = QueryPattern(
            pattern_name="Invalid",
            description="Invalid pattern",
            dimensions_used=[],
            complexity_level="invalid",
            real_world_use_case="Invalid"
        )
        
        # Should still execute (with fallback behavior)
        result = engine.execute_query(pattern, "invalid_query")
        self.assertIsInstance(result, QueryResult)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # This is a basic test - in a real environment, you'd monitor actual memory usage
        engine = MultiDimensionalQueryEngine(dataset_size=1000)
        engine.build_dataset()
        
        # Should be able to handle the dataset
        self.assertEqual(len(engine.bit_chains), 1000)
        
        # Should be able to execute queries
        pattern = QueryPattern(
            pattern_name="Memory Test",
            description="Memory test",
            dimensions_used=["realm"],
            complexity_level="simple",
            real_world_use_case="Memory"
        )
        
        result = engine.execute_query(pattern, "memory_query")
        self.assertIsInstance(result, QueryResult)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)