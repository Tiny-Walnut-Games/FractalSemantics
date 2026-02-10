#!/usr/bin/env python3
"""
Parameter Sweep Runner for FractalSemantics Experiments

This script systematically tests different parameter combinations for EXP-08, EXP-09, and EXP-10
to identify optimal parameter ranges for achieving passing scores.
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import experiments directly to avoid circular imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fractalsemantics'))

from fractalsemantics.exp08_self_organizing_memory import SelfOrganizingMemoryExperiment, SelfOrganizingMemoryResults
from fractalsemantics.exp09_memory_pressure import MemoryPressureExperiment, MemoryPressureResults
from fractalsemantics.exp10_multidimensional_query import MultiDimensionalQueryExperiment, MultiDimensionalQueryResults


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweep testing."""
    
    experiment_name: str
    parameter_name: str
    parameter_values: List[Any]
    baseline_value: Any
    success_criteria: Dict[str, float]
    timeout_seconds: int = 600  # 10 minutes max per test


@dataclass
class SweepResult:
    """Result from a single parameter sweep test."""
    
    experiment_name: str
    parameter_name: str
    parameter_value: Any
    status: str
    metrics: Dict[str, float]
    execution_time_seconds: float
    timestamp: str


class ParameterSweepRunner:
    """Runner for systematic parameter sweeps across experiments."""
    
    def __init__(self):
        self.results: List[SweepResult] = []
        self.sweep_configs = self._define_sweep_configs()
    
    def _define_sweep_configs(self) -> List[ParameterSweepConfig]:
        """Define parameter sweep configurations for all experiments."""
        
        configs = []
        
        # EXP-08: Self-Organizing Memory Networks
        configs.append(ParameterSweepConfig(
            experiment_name="EXP-08",
            parameter_name="num_memories",
            parameter_values=[100, 250, 500, 1000, 2000, 5000],
            baseline_value=1000,
            success_criteria={
                "semantic_cohesion_score": 0.8,
                "retrieval_efficiency": 0.8,
                "storage_overhead_reduction": 0.5,
                "emergent_intelligence_score": 0.6
            }
        ))
        
        configs.append(ParameterSweepConfig(
            experiment_name="EXP-08",
            parameter_name="consolidation_threshold",
            parameter_values=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            baseline_value=0.8,
            success_criteria={
                "semantic_cohesion_score": 0.8,
                "retrieval_efficiency": 0.8,
                "storage_overhead_reduction": 0.5,
                "emergent_intelligence_score": 0.6
            }
        ))
        
        # EXP-09: Memory Pressure Testing
        configs.append(ParameterSweepConfig(
            experiment_name="EXP-09",
            parameter_name="max_memory_target_mb",
            parameter_values=[100, 200, 500, 1000, 2000, 5000],
            baseline_value=1000,
            success_criteria={
                "degradation_ratio": 10.0,  # Lower is better
                "stability_score": 0.6,
                "graceful_degradation": 1.0,  # Boolean
                "optimization_improvement": 0.3
            }
        ))
        
        # EXP-10: Multi-Dimensional Query Optimization
        configs.append(ParameterSweepConfig(
            experiment_name="EXP-10",
            parameter_name="dataset_size",
            parameter_values=[1000, 2500, 5000, 10000, 25000, 50000],
            baseline_value=10000,
            success_criteria={
                "avg_query_time_ms": 100.0,  # Lower is better
                "avg_f1_score": 0.75,
                "query_throughput_qps": 10.0,
                "practical_value_score": 0.6
            }
        ))
        
        return configs
    
    def run_sweep(self, config: ParameterSweepConfig) -> List[SweepResult]:
        """Run parameter sweep for a specific configuration."""
        print(f"\n{'='*80}")
        print(f"RUNNING PARAMETER SWEEP: {config.experiment_name} - {config.parameter_name}")
        print(f"{'='*80}")
        print(f"Parameter values: {config.parameter_values}")
        print(f"Success criteria: {config.success_criteria}")
        
        results = []
        
        for value in config.parameter_values:
            print(f"\nTesting {config.parameter_name} = {value}")
            
            try:
                result = self._run_single_test(config, value)
                results.append(result)
                self.results.append(result)
                
                print(f"  Status: {result.status}")
                print(f"  Execution time: {result.execution_time_seconds:.2f}s")
                print(f"  Key metrics: {self._format_metrics(result.metrics)}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                # Create failed result
                failed_result = SweepResult(
                    experiment_name=config.experiment_name,
                    parameter_name=config.parameter_name,
                    parameter_value=value,
                    status="ERROR",
                    metrics={},
                    execution_time_seconds=0.0,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                results.append(failed_result)
                self.results.append(failed_result)
        
        return results
    
    def _run_single_test(self, config: ParameterSweepConfig, value: Any) -> SweepResult:
        """Run a single test with the specified parameter value."""
        start_time = time.time()
        
        if config.experiment_name == "EXP-08":
            result = self._run_exp08_test(value, config.parameter_name)
        elif config.experiment_name == "EXP-09":
            result = self._run_exp09_test(value, config.parameter_name)
        elif config.experiment_name == "EXP-10":
            result = self._run_exp10_test(value, config.parameter_name)
        else:
            raise ValueError(f"Unknown experiment: {config.experiment_name}")
        
        execution_time = time.time() - start_time
        
        return SweepResult(
            experiment_name=config.experiment_name,
            parameter_name=config.parameter_name,
            parameter_value=value,
            status=result.status,
            metrics=self._extract_metrics(result),
            execution_time_seconds=execution_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def _run_exp08_test(self, value: Any, parameter_name: str) -> SelfOrganizingMemoryResults:
        """Run EXP-08 with specified parameter."""
        if parameter_name == "num_memories":
            experiment = SelfOrganizingMemoryExperiment(
                num_memories=value,
                consolidation_threshold=0.8  # Use baseline
            )
        elif parameter_name == "consolidation_threshold":
            experiment = SelfOrganizingMemoryExperiment(
                num_memories=1000,  # Use baseline
                consolidation_threshold=value
            )
        else:
            raise ValueError(f"Unknown EXP-08 parameter: {parameter_name}")
        
        return experiment.run()
    
    def _run_exp09_test(self, value: Any, parameter_name: str) -> MemoryPressureResults:
        """Run EXP-09 with specified parameter."""
        if parameter_name == "max_memory_target_mb":
            experiment = MemoryPressureExperiment(max_memory_target_mb=value)
            return experiment.run()
        else:
            raise ValueError(f"Unknown EXP-09 parameter: {parameter_name}")
    
    def _run_exp10_test(self, value: Any, parameter_name: str) -> MultiDimensionalQueryResults:
        """Run EXP-10 with specified parameter."""
        if parameter_name == "dataset_size":
            experiment = MultiDimensionalQueryExperiment(dataset_size=value)
            return experiment.run()
        else:
            raise ValueError(f"Unknown EXP-10 parameter: {parameter_name}")
    
    def _extract_metrics(self, result: Any) -> Dict[str, float]:
        """Extract key metrics from experiment result."""
        if isinstance(result, SelfOrganizingMemoryResults):
            return {
                "semantic_cohesion_score": result.semantic_cohesion_score,
                "retrieval_efficiency": result.retrieval_efficiency,
                "storage_overhead_reduction": result.storage_overhead_reduction,
                "emergent_intelligence_score": result.emergent_intelligence_score,
                "num_clusters": result.num_clusters,
                "avg_cluster_size": result.avg_cluster_size
            }
        elif isinstance(result, MemoryPressureResults):
            return {
                "degradation_ratio": result.degradation_ratio,
                "stability_score": result.stability_score,
                "graceful_degradation": 1.0 if result.graceful_degradation else 0.0,
                "optimization_improvement": result.optimization_improvement,
                "peak_memory_usage_mb": result.peak_memory_usage_mb,
                "memory_efficiency_score": result.memory_efficiency_score
            }
        elif isinstance(result, MultiDimensionalQueryResults):
            return {
                "avg_query_time_ms": result.avg_query_time_ms,
                "avg_f1_score": result.avg_f1_score,
                "query_throughput_qps": result.query_throughput_qps,
                "practical_value_score": result.practical_value_score,
                "scalability_score": result.scalability_score,
                "optimization_improvement": result.optimization_improvement
            }
        else:
            return {}
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display."""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.3f}")
            else:
                formatted.append(f"{key}: {value}")
        return ", ".join(formatted)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze sweep results and identify optimal parameters."""
        print(f"\n{'='*80}")
        print("PARAMETER SWEEP ANALYSIS")
        print(f"{'='*80}")
        
        analysis = {}
        
        # Group results by experiment and parameter
        experiment_groups = {}
        for result in self.results:
            key = f"{result.experiment_name}_{result.parameter_name}"
            if key not in experiment_groups:
                experiment_groups[key] = []
            experiment_groups[key].append(result)
        
        for group_key, group_results in experiment_groups.items():
            print(f"\nAnalyzing {group_key}:")
            
            # Sort by parameter value
            group_results.sort(key=lambda r: r.parameter_value)
            
            # Calculate success rates
            pass_count = sum(1 for r in group_results if r.status == "PASS")
            total_count = len(group_results)
            success_rate = pass_count / total_count if total_count > 0 else 0.0
            
            print(f"  Success rate: {success_rate:.1%} ({pass_count}/{total_count})")
            
            # Find optimal parameters
            optimal_params = self._find_optimal_parameters(group_results)
            
            # Find minimum viable parameters
            min_viable_params = self._find_minimum_viable_parameters(group_results)
            
            # Find overkill thresholds
            overkill_thresholds = self._find_overkill_thresholds(group_results)
            
            analysis[group_key] = {
                "success_rate": success_rate,
                "optimal_parameters": optimal_params,
                "minimum_viable_parameters": min_viable_params,
                "overkill_thresholds": overkill_thresholds,
                "results": [asdict(r) for r in group_results]
            }
            
            print(f"  Optimal parameters: {optimal_params}")
            print(f"  Minimum viable: {min_viable_params}")
            print(f"  Overkill thresholds: {overkill_thresholds}")
        
        return analysis
    
    def _find_optimal_parameters(self, results: List[SweepResult]) -> Dict[str, Any]:
        """Find optimal parameter values based on performance."""
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if r.status == "PASS"]
        if not successful_results:
            return {}
        
        # Find best performing result
        best_result = max(successful_results, key=lambda r: self._calculate_performance_score(r))
        
        return {
            "value": best_result.parameter_value,
            "performance_score": self._calculate_performance_score(best_result),
            "status": best_result.status
        }
    
    def _find_minimum_viable_parameters(self, results: List[SweepResult]) -> Dict[str, Any]:
        """Find minimum parameter values that achieve passing scores."""
        successful_results = [r for r in results if r.status == "PASS"]
        if not successful_results:
            return {}
        
        # Find minimum value that passes
        min_successful = min(successful_results, key=lambda r: r.parameter_value)
        
        return {
            "value": min_successful.parameter_value,
            "performance_score": self._calculate_performance_score(min_successful),
            "status": min_successful.status
        }
    
    def _find_overkill_thresholds(self, results: List[SweepResult]) -> Dict[str, Any]:
        """Find parameter values where additional resources provide diminishing returns."""
        if len(results) < 3:
            return {}
        
        # Sort by parameter value
        results.sort(key=lambda r: r.parameter_value)
        
        # Calculate performance improvements
        improvements = []
        for i in range(1, len(results)):
            prev_score = self._calculate_performance_score(results[i-1])
            curr_score = self._calculate_performance_score(results[i])
            
            if prev_score > 0:
                improvement = (curr_score - prev_score) / prev_score
                improvements.append({
                    "from": results[i-1].parameter_value,
                    "to": results[i].parameter_value,
                    "improvement": improvement
                })
        
        # Find where improvements drop below 5%
        overkill_thresholds = []
        for improvement in improvements:
            if improvement["improvement"] < 0.05:  # Less than 5% improvement
                overkill_thresholds.append({
                    "threshold": improvement["to"],
                    "improvement": improvement["improvement"]
                })
        
        if overkill_thresholds:
            return {
                "value": overkill_thresholds[0]["threshold"],
                "improvement": overkill_thresholds[0]["improvement"]
            }
        
        return {}
    
    def _calculate_performance_score(self, result: SweepResult) -> float:
        """Calculate overall performance score for a result."""
        if not result.metrics:
            return 0.0
        
        # Weight different metrics based on importance
        weights = {
            "semantic_cohesion_score": 0.25,
            "retrieval_efficiency": 0.25,
            "storage_overhead_reduction": 0.15,
            "emergent_intelligence_score": 0.15,
            "degradation_ratio": -0.1,  # Lower is better
            "stability_score": 0.2,
            "avg_query_time_ms": -0.15,  # Lower is better
            "avg_f1_score": 0.25,
            "query_throughput_qps": 0.15,
            "practical_value_score": 0.2
        }
        
        score = 0.0
        for metric, value in result.metrics.items():
            if metric in weights:
                weight = weights[metric]
                if weight > 0:
                    score += value * weight
                else:
                    # For metrics where lower is better
                    score += (1.0 - min(1.0, value / 100.0)) * abs(weight)
        
        return score
    
    def save_results(self, output_file: str = None) -> str:
        """Save all results to JSON file."""
        if output_file is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = f"parameter_sweep_results_{timestamp}.json"
        
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = str(results_dir / output_file)
        
        # Save detailed results
        with open(output_path, "w", encoding="UTF-8") as f:
            json.dump({
                "sweep_results": [asdict(r) for r in self.results],
                "analysis": self.analyze_results(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path


def main():
    """Main entry point for parameter sweep runner."""
    print("FractalSemantics Parameter Sweep Runner")
    print("=" * 50)
    
    runner = ParameterSweepRunner()
    
    # Run sweeps for all configurations
    for config in runner.sweep_configs:
        runner.run_sweep(config)
    
    # Analyze results
    runner.analyze_results()
    
    # Save results
    output_file = runner.save_results()
    
    print(f"\n{'='*80}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total tests executed: {len(runner.results)}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
