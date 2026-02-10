#!/usr/bin/env python3
"""
Experiment Runner for FractalSemantics HTML Web Application

This script provides the backend execution capabilities for the HTML web application,
allowing it to run real FractalSemantics experiments with educational output.
"""

import asyncio
import json
import sys
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent / 'fractalsemantics'))

@dataclass
class ExperimentResult:
    """Result of an experiment execution."""
    experiment_id: str
    success: bool
    duration: float
    output: str
    metrics: Dict[str, Any]
    educational_content: List[str]

class ExperimentRunner:
    """Runs FractalSemantics experiments with educational output."""
    
    def __init__(self):
        self.experiment_configs = {
            "EXP-01": {
                "module": "fractalsemantics.exp01_geometric_collision",
                "description": "Tests that every bit-chain gets a unique address with zero collisions using 8-dimensional coordinates.",
                "educational_focus": "8-Dimensional Coordinate Space and Collision Resistance Mathematics"
            },
            "EXP-02": {
                "module": "fractalsemantics.exp02_retrieval_efficiency", 
                "description": "Tests sub-millisecond retrieval performance at scale using hash table indexing.",
                "educational_focus": "Hash Table Performance Analysis and Big O Notation"
            },
            "EXP-03": {
                "module": "fractalsemantics.exp03_coordinate_entropy",
                "description": "Validates that all 7 dimensions are necessary to avoid collisions through ablation testing.",
                "educational_focus": "Dimensional Analysis and Shannon Entropy Calculation"
            },
            "EXP-04": {
                "module": "fractalsemantics.exp04_fractal_scaling",
                "description": "Tests consistency of addressing properties across different scales (1K to 1M entities).",
                "educational_focus": "Fractal Geometry Principles and Scale Invariance Analysis"
            },
            "EXP-05": {
                "module": "fractalsemantics.exp05_compression_expansion",
                "description": "Tests lossless compression through hierarchical structures (fragments → clusters → glyphs → mist).",
                "educational_focus": "Information Theory and Hierarchical Compression Algorithms"
            },
            "EXP-06": {
                "module": "fractalsemantics.exp06_entanglement_detection",
                "description": "Tests detection of narrative entanglement between bit-chains using semantic similarity.",
                "educational_focus": "Semantic Similarity Metrics and Cosine Similarity Calculation"
            },
            "EXP-07": {
                "module": "fractalsemantics.exp07_luca_bootstrap",
                "description": "Tests bootstrapping from Last Universal Common Ancestor to derive all entities.",
                "educational_focus": "Evolutionary Algorithms and Lineage Tree Generation"
            },
            "EXP-08": {
                "module": "fractalsemantics.exp08_self_organizing_memory",
                "description": "Tests FractalSemantics's ability to create self-organizing memory structures with semantic clustering.",
                "educational_focus": "Neural Network Clustering and Self-Organization Principles"
            },
            "EXP-09": {
                "module": "fractalsemantics.exp09_memory_pressure",
                "description": "Tests system resilience and performance under constrained memory conditions.",
                "educational_focus": "Memory Management Algorithms and Performance Under Constraints"
            },
            "EXP-10": {
                "module": "fractalsemantics.exp10_multidimensional_query",
                "description": "Tests FractalSemantics's unique querying capabilities across all 8 dimensions.",
                "educational_focus": "Multi-Dimensional Indexing and Query Optimization Algorithms"
            },
            "EXP-11": {
                "module": "fractalsemantics.exp11_dimension_cardinality",
                "description": "Explores pros and cons of 7 dimensions vs. more or fewer dimensions.",
                "educational_focus": "Dimensional Trade-off Analysis and Optimal Dimension Count"
            },
            "EXP-12": {
                "module": "fractalsemantics.exp12_benchmark_comparison",
                "description": "Compares FractalSemantics against common systems (UUID, SHA256, Vector DB, etc.).",
                "educational_focus": "Comparative Performance Analysis and Benchmarking Methodologies"
            }
        }

    async def run_experiment(self, experiment_id: str, quick_mode: bool = False) -> ExperimentResult:
        """Run a single experiment with educational output."""
        if experiment_id not in self.experiment_configs:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        
        config = self.experiment_configs[experiment_id]
        start_time = time.time()
        
        educational_content = []
        
        try:
            # Generate educational introduction
            educational_content.append(self._generate_introduction(experiment_id, config))
            
            # Run the actual experiment
            result = await self._execute_experiment_module(experiment_id, quick_mode)
            
            # Add educational analysis
            educational_content.append(self._generate_analysis(experiment_id, result))
            
            duration = time.time() - start_time
            
            return ExperimentResult(
                experiment_id=experiment_id,
                success=result["success"],
                duration=duration,
                output=result["output"],
                metrics=result["metrics"],
                educational_content=educational_content
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                duration=duration,
                output=f"Error executing experiment: {str(e)}",
                metrics={},
                educational_content=[f"❌ Experiment failed with error: {str(e)}"]
            )

    def _generate_introduction(self, experiment_id: str, config: Dict[str, Any]) -> str:
        """Generate educational introduction for the experiment."""
        intro = f"""
🎓 EXPERIMENT: {experiment_id} - {config['module'].split('.')[-1].replace('_', ' ').title()}
📚 Educational Focus: {config['educational_focus']}

🎯 Objective:
{config['description']}

📝 Mathematical Concepts Covered:
"""
        
        # Add specific mathematical concepts for each experiment
        concepts = self._get_mathematical_concepts(experiment_id)
        for concept in concepts:
            intro += f"   • {concept}\n"
        
        intro += """
🔍 Step-by-Step Process:
"""
        
        # Add step-by-step process
        steps = self._get_experiment_steps(experiment_id)
        for i, step in enumerate(steps, 1):
            intro += f"   {i}. {step}\n"
        
        intro += "\n" + "="*60 + "\n"
        return intro

    def _generate_analysis(self, experiment_id: str, result: Dict[str, Any]) -> str:
        """Generate educational analysis of experiment results."""
        analysis = f"""
📊 EXPERIMENT RESULTS ANALYSIS: {experiment_id}
🎯 Key Learning Outcomes:

"""
        
        if result["success"]:
            analysis += "✅ Experiment completed successfully!\n\n"
            analysis += "📈 Performance Metrics:\n"
            for key, value in result["metrics"].items():
                analysis += f"   • {key}: {value}\n"
        else:
            analysis += "❌ Experiment encountered issues.\n\n"
            analysis += "🔍 Troubleshooting Insights:\n"
            analysis += "   • This demonstrates real-world challenges in computational systems\n"
            analysis += "   • Error analysis helps identify system limitations\n"
            analysis += "   • Understanding failure modes is crucial for system design\n"
        
        analysis += f"""
💡 Real-World Applications:
{self._get_real_world_applications(experiment_id)}

🎯 Takeaway Lessons:
{self._get_key_lessons(experiment_id)}
"""
        
        return analysis

    def _get_mathematical_concepts(self, experiment_id: str) -> List[str]:
        """Get mathematical concepts for the experiment."""
        concepts_map = {
            "EXP-01": [
                "8-Dimensional Coordinate Space",
                "Collision Resistance Mathematics", 
                "Address Generation Formula",
                "Geometric Probability Theory"
            ],
            "EXP-02": [
                "Hash Table Performance Analysis",
                "Big O Notation (O(1) retrieval)",
                "Latency Measurement Statistics",
                "Time Complexity Analysis"
            ],
            "EXP-03": [
                "Dimensional Analysis",
                "Shannon Entropy Calculation",
                "Ablation Study Methodology",
                "Information Theory Fundamentals"
            ],
            "EXP-04": [
                "Fractal Geometry Principles",
                "Scale Invariance Analysis",
                "Power Law Distributions",
                "Self-Similarity Mathematics"
            ],
            "EXP-05": [
                "Information Theory",
                "Huffman Coding Principles",
                "Hierarchical Compression Algorithms",
                "Lossless Compression Mathematics"
            ],
            "EXP-06": [
                "Semantic Similarity Metrics",
                "Cosine Similarity Calculation",
                "Entanglement Threshold Analysis",
                "Vector Space Mathematics"
            ],
            "EXP-07": [
                "Evolutionary Algorithms",
                "Lineage Tree Generation",
                "Genetic Distance Metrics",
                "Phylogenetic Analysis"
            ],
            "EXP-08": [
                "Neural Network Clustering",
                "Self-Organization Principles",
                "Semantic Distance Metrics",
                "Network Topology Analysis"
            ],
            "EXP-09": [
                "Memory Management Algorithms",
                "Performance Under Constraints",
                "Resource Optimization",
                "System Resilience Analysis"
            ],
            "EXP-10": [
                "Multi-Dimensional Indexing",
                "Query Optimization Algorithms",
                "Dimensional Pruning Strategies",
                "Spatial Database Theory"
            ],
            "EXP-11": [
                "Dimensional Trade-off Analysis",
                "Expressiveness vs. Complexity",
                "Optimal Dimension Count",
                "Pareto Efficiency Analysis"
            ],
            "EXP-12": [
                "Comparative Performance Analysis",
                "Benchmarking Methodologies",
                "System Trade-off Evaluation",
                "Statistical Significance Testing"
            ]
        }
        return concepts_map.get(experiment_id, ["General Computational Concepts"])

    def _get_experiment_steps(self, experiment_id: str) -> List[str]:
        """Get step-by-step process for the experiment."""
        steps_map = {
            "EXP-01": [
                "Generate random bit-chains with specified sample size",
                "Compute FractalSemantics coordinates for each bit-chain",
                "Calculate unique addresses using coordinate hashing",
                "Verify zero collisions across all generated addresses",
                "Analyze distribution patterns and statistical properties"
            ],
            "EXP-02": [
                "Build hash table index mapping addresses to bit-chains",
                "Generate random retrieval queries across the dataset",
                "Measure query response times with high-precision timing",
                "Calculate average and percentile latencies",
                "Verify sub-millisecond performance requirements"
            ],
            "EXP-03": [
                "Calculate baseline entropy with complete 7-dimensional coordinates",
                "Remove each dimension individually through ablation",
                "Measure entropy reduction for each dimension removal",
                "Identify critical dimensions that significantly impact entropy",
                "Validate necessity threshold for collision avoidance"
            ],
            "EXP-04": [
                "Generate datasets at multiple scales (1K, 10K, 100K, 1M entities)",
                "Measure collision rates at each scale",
                "Analyze retrieval performance scaling characteristics",
                "Verify fractal properties and self-similarity",
                "Calculate scaling exponents and power law relationships"
            ],
            "EXP-05": [
                "Create hierarchical data structures (fragments → clusters → glyphs → mist)",
                "Apply compression algorithms at each hierarchical level",
                "Measure compression ratios and efficiency metrics",
                "Verify lossless decompression capabilities",
                "Analyze compression effectiveness across different data types"
            ],
            "EXP-06": [
                "Generate related bit-chain pairs with known semantic relationships",
                "Calculate semantic similarity scores using vector embeddings",
                "Apply entanglement detection algorithm with configurable thresholds",
                "Measure precision and recall of entanglement detection",
                "Validate threshold effectiveness across different similarity levels"
            ],
            "EXP-07": [
                "Define LUCA (Last Universal Common Ancestor) entity with base coordinates",
                "Generate evolutionary tree through lineage operations",
                "Calculate lineage relationships and genetic distances",
                "Verify bootstrap completeness and coverage",
                "Analyze genetic diversity and evolutionary patterns"
            ],
            "EXP-08": [
                "Generate memory network with semantic content",
                "Apply clustering algorithms based on semantic similarity",
                "Measure semantic coherence within clusters",
                "Evaluate self-organization and emergent structure",
                "Analyze network topology and connectivity patterns"
            ],
            "EXP-09": [
                "Establish baseline performance metrics under normal conditions",
                "Apply memory pressure scenarios (light, moderate, heavy, critical)",
                "Measure performance degradation under constrained conditions",
                "Test optimization strategies (lazy loading, compression, eviction)",
                "Analyze system resilience and recovery characteristics"
            ],
            "EXP-10": [
                "Create multi-dimensional index structures for efficient querying",
                "Generate complex query patterns across all 8 dimensions",
                "Measure query execution times and resource usage",
                "Apply optimization techniques (indexing, caching, pruning)",
                "Analyze dimensional pruning effectiveness and query complexity"
            ],
            "EXP-11": [
                "Test various dimension counts (3, 4, 5, 6, 7, 8, 9, 10 dimensions)",
                "Measure expressiveness scores for each dimension count",
                "Calculate complexity overhead and computational cost",
                "Find optimal balance between expressiveness and efficiency",
                "Validate theoretical predictions with empirical results"
            ],
            "EXP-12": [
                "Define comprehensive comparison metrics (performance, storage, expressiveness)",
                "Test all benchmark systems (UUID, SHA256, Vector DB, Graph DB, RDBMS)",
                "Measure performance characteristics across different scales",
                "Calculate relative advantages and disadvantages",
                "Analyze trade-offs and identify optimal use cases for each system"
            ]
        }
        return steps_map.get(experiment_id, ["Execute experiment", "Analyze results", "Generate report"])

    def _get_real_world_applications(self, experiment_id: str) -> str:
        """Get real-world applications for the experiment."""
        applications_map = {
            "EXP-01": "• Content-addressable storage systems\n• Cryptographic hash functions\n• Database indexing strategies\n• File system design",
            "EXP-02": "• Database query optimization\n• Cache system design\n• Real-time data processing\n• High-frequency trading systems",
            "EXP-03": "• Feature selection in machine learning\n• Dimensionality reduction techniques\n• Data compression algorithms\n• Information retrieval systems",
            "EXP-04": "• Scalable distributed systems\n• Cloud computing architectures\n• Big data processing frameworks\n• Network protocol design",
            "EXP-05": "• Data compression software\n• Multimedia file formats\n• Database storage optimization\n• Network bandwidth optimization",
            "EXP-06": "• Semantic search engines\n• Recommendation systems\n• Natural language processing\n• Knowledge graph construction",
            "EXP-07": "• Evolutionary biology research\n• Phylogenetic tree construction\n• Genetic algorithm design\n• Ancestral sequence reconstruction",
            "EXP-08": "• Artificial neural networks\n• Knowledge management systems\n• Self-organizing maps\n• Clustering algorithms",
            "EXP-09": "• Memory-constrained embedded systems\n• Mobile application optimization\n• Cloud resource management\n• Real-time system design",
            "EXP-10": "• Multi-dimensional database systems\n• Geographic information systems\n• Scientific data analysis\n• Complex query optimization",
            "EXP-11": "• System design trade-off analysis\n• Resource allocation strategies\n• Performance optimization\n• Cost-benefit analysis",
            "EXP-12": "• Technology selection for projects\n• Performance benchmarking\n• System architecture design\n• Vendor evaluation"
        }
        return applications_map.get(experiment_id, "• General computational applications")

    def _get_key_lessons(self, experiment_id: str) -> str:
        """Get key lessons for the experiment."""
        lessons_map = {
            "EXP-01": "• Mathematical foundations ensure system reliability\n• Collision resistance is critical for data integrity\n• Proper coordinate systems enable unique addressing\n• Cryptographic principles provide security guarantees",
            "EXP-02": "• Algorithmic efficiency impacts real-world performance\n• Hash tables provide optimal retrieval performance\n• System design must consider scalability\n• Performance measurement requires precise timing",
            "EXP-03": "• Dimensional analysis reveals system properties\n• Information theory guides feature selection\n• Ablation studies identify critical components\n• Entropy measures system complexity",
            "EXP-04": "• Fractal properties enable scalable systems\n• Self-similarity provides consistent behavior\n• Scale invariance ensures predictable performance\n• Power laws describe natural system behavior",
            "EXP-05": "• Hierarchical structures enable efficient compression\n• Information theory guides algorithm design\n• Lossless compression preserves data integrity\n• Multi-level optimization improves efficiency",
            "EXP-06": "• Semantic similarity enables intelligent systems\n• Vector embeddings capture meaningful relationships\n• Threshold selection balances precision and recall\n• Entanglement detection reveals hidden connections",
            "EXP-07": "• Evolutionary principles guide system design\n• Lineage tracking enables provenance\n• Bootstrap methods create comprehensive systems\n• Genetic algorithms solve complex problems",
            "EXP-08": "• Self-organization creates emergent intelligence\n• Clustering reveals natural data structure\n• Semantic coherence improves system usability\n• Network topology affects performance",
            "EXP-09": "• Resource constraints drive innovation\n• Optimization strategies improve resilience\n• Performance under pressure reveals system quality\n• Memory management is critical for efficiency",
            "EXP-10": "• Multi-dimensional indexing enables complex queries\n• Query optimization reduces computational complexity\n• Dimensional pruning improves performance\n• Spatial databases handle complex data relationships",
            "EXP-11": "• Trade-off analysis guides system design\n• Optimal dimensionality balances expressiveness and complexity\n• Pareto efficiency identifies best solutions\n• Complexity theory informs algorithm selection",
            "EXP-12": "• Comparative analysis reveals system strengths\n• Benchmarking provides objective evaluation\n• Performance metrics guide technology selection\n• Trade-off analysis informs architectural decisions"
        }
        return lessons_map.get(experiment_id, "• Computational thinking solves complex problems\n• Mathematical foundations enable reliable systems\n• Experimental methodology validates theoretical concepts")

    async def _execute_experiment_module(self, experiment_id: str, quick_mode: bool) -> Dict[str, Any]:
        """Execute the actual experiment module."""
        
        try:
            # Try to import and run the experiment module
            if experiment_id == "EXP-01":
                from fractalsemantics.exp01_geometric_collision import EXP01_GeometricCollision
                exp = EXP01_GeometricCollision()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-02":
                from fractalsemantics.exp02_retrieval_efficiency import EXP02_RetrievalEfficiency
                exp = EXP02_RetrievalEfficiency()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-03":
                from fractalsemantics.exp03_coordinate_entropy import EXP03_CoordinateEntropy
                exp = EXP03_CoordinateEntropy()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-04":
                from fractalsemantics.exp04_fractal_scaling import EXP04_FractalScaling
                exp = EXP04_FractalScaling()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-05":
                from fractalsemantics.exp05_compression_expansion import EXP05_CompressionExpansion
                exp = EXP05_CompressionExpansion()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-06":
                from fractalsemantics.exp06_entanglement_detection import EXP06_EntanglementDetection
                exp = EXP06_EntanglementDetection()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-07":
                from fractalsemantics.exp07_luca_bootstrap import EXP07_LUCABootstrap
                exp = EXP07_LUCABootstrap()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-08":
                from fractalsemantics.exp08_self_organizing_memory import EXP08_SelfOrganizingMemory
                exp = EXP08_SelfOrganizingMemory()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-09":
                from fractalsemantics.exp09_memory_pressure import EXP09_MemoryPressure
                exp = EXP09_MemoryPressure()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-10":
                from fractalsemantics.exp10_multidimensional_query import EXP10_MultiDimensionalQuery
                exp = EXP10_MultiDimensionalQuery()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-11":
                from fractalsemantics.exp11_dimension_cardinality import EXP11_DimensionCardinality
                exp = EXP11_DimensionCardinality()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            elif experiment_id == "EXP-12":
                from fractalsemantics.exp12_benchmark_comparison import EXP12_BenchmarkComparison
                exp = EXP12_BenchmarkComparison()
                result = exp.run_experiment(quick_mode=quick_mode)
                
            else:
                raise ValueError(f"No specific handler for {experiment_id}")
            
            return {
                "success": True,
                "output": f"Experiment {experiment_id} completed successfully",
                "metrics": result.get("metrics", {})
            }
            
        except ImportError:
            # Fallback to subprocess execution
            return await self._execute_experiment_subprocess(experiment_id, quick_mode)
        except Exception as e:
            return {
                "success": False,
                "output": f"Error in experiment execution: {str(e)}",
                "metrics": {}
            }

    async def _execute_experiment_subprocess(self, experiment_id: str, quick_mode: bool) -> Dict[str, Any]:
        """Execute experiment as subprocess."""
        try:
            # Construct command to run the experiment
            cmd = [
                sys.executable, "-m", "fractalsemantics.fractalsemantics_experiments",
                "--experiment", experiment_id
            ]
            
            if quick_mode:
                cmd.append("--quick")
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            output = stdout.decode('utf-8')
            error = stderr.decode('utf-8')
            
            success = process.returncode == 0
            
            return {
                "success": success,
                "output": output + (f"\nError: {error}" if error else ""),
                "metrics": {"return_code": process.returncode}
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Subprocess execution failed: {str(e)}",
                "metrics": {}
            }

async def main():
    """Main entry point for the experiment runner."""
    if len(sys.argv) < 2:
        print("Usage: python experiment_runner.py <experiment_id> [--quick]")
        sys.exit(1)
    
    experiment_id = sys.argv[1]
    quick_mode = "--quick" in sys.argv
    
    runner = ExperimentRunner()
    
    try:
        result = await runner.run_experiment(experiment_id, quick_mode)
        
        # Output result as JSON for the HTML app
        output = {
            "experiment_id": result.experiment_id,
            "success": result.success,
            "duration": result.duration,
            "output": result.output,
            "metrics": result.metrics,
            "educational_content": result.educational_content
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "success": False
        }, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
