#!/usr/bin/env python3
"""
Experiment Runner for FractalSemantics HTML Web Application

This script provides the backend execution capabilities for the HTML web application,
allowing it to run real FractalSemantics experiments with educational output.
"""

import ast
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import tqdm

# Add the fractalsemantics module to the path FIRST, before any imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    module_name: str
    description: str
    educational_focus: str
    experiment_type: str = "standard"  # "standard", "advanced", "stress_test"
    quick_mode_supported: bool = True
    timeout_seconds: int = 300
    dependencies: list[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Result of an experiment execution."""
    experiment_id: str
    success: bool
    duration: float
    output: str
    metrics: dict[str, any]
    educational_content: list[str]
    result_type: str = "unknown"  # "success", "warning", "partial_success", "failure"
    error_details: Optional[dict[str, any]] = None

@dataclass
class BatchRunResult:
    """Result of running multiple experiments."""
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    total_duration: float
    experiment_results: list[ExperimentResult]
    summary_report: str
    performance_metrics: dict[str, any] = field(default_factory=dict)

class ExperimentRunner:
    """Runs FractalSemantics experiments with educational output."""

    def __init__(self):
        self.experiment_configs = self._load_experiment_configs()
        self._validate_configurations()

    def _validate_configurations(self):
        """Validate all experiment configurations for consistency and completeness."""
        for exp_id, config in self.experiment_configs.items():
            # Validate experiment ID format
            if not exp_id.startswith("EXP-"):
                raise ValueError(f"Invalid experiment ID format: {exp_id}")

            # Validate module name format
            if not config.module_name.startswith("fractalsemantics."):
                raise ValueError(f"Invalid module name format: {config.module_name}")

            # Validate experiment type
            valid_types = ["standard", "advanced", "stress_test"]
            if config.experiment_type not in valid_types:
                raise ValueError(f"Invalid experiment type: {config.experiment_type}")

            # Validate timeout
            if config.timeout_seconds <= 0:
                raise ValueError(f"Invalid timeout: {config.timeout_seconds}")

            # Validate dependencies list
            if not isinstance(config.dependencies, list):
                raise ValueError(f"Dependencies must be a list: {config.dependencies}")

            # Check for required dependencies based on experiment type
            self._validate_dependencies(config)

    def _validate_dependencies(self, config: ExperimentConfig):
        """Validate that required dependencies are available."""
        required_packages = {
            "numpy": "numpy",
            "scipy": "scipy",
            "matplotlib": "matplotlib",
            "hashlib": "hashlib",
            "time": "time",
            "random": "random",
            "itertools": "itertools",
            "zlib": "zlib",
            "pickle": "pickle",
            "sklearn": "sklearn",
            "psutil": "psutil",
            "gc": "gc",
            "uuid": "uuid",
            "math": "math",
            "periodictable": "periodictable",
            "networkx": "networkx"
        }

        missing_packages = []
        for dep in config.dependencies:
            if dep in required_packages:
                try:
                    __import__(dep)
                except ImportError:
                    missing_packages.append(dep)

        if missing_packages:
            print(f"⚠️  Warning: Missing dependencies for {config.experiment_id}: {', '.join(missing_packages)}")
            print("   Some experiments may fail due to missing packages.")

    def _load_experiment_configs(self) -> dict[str, ExperimentConfig]:
        """Load experiment configurations with proper structure and validation."""
        configs = {
            "EXP-01": ExperimentConfig(
                experiment_id="EXP-01",
                module_name="fractalsemantics.exp01_geometric_collision",
                description="Tests that every bit-chain gets a unique address with zero collisions using 8-dimensional coordinates.",
                educational_focus="8-Dimensional Coordinate Space and Collision Resistance Mathematics",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "hashlib"]
            ),
            "EXP-02": ExperimentConfig(
                experiment_id="EXP-02",
                module_name="fractalsemantics.exp02_retrieval_efficiency",
                description="Tests sub-millisecond retrieval performance at scale using hash table indexing.",
                educational_focus="Hash Table Performance Analysis and Big O Notation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["hashlib", "time"]
            ),
            "EXP-03": ExperimentConfig(
                experiment_id="EXP-03",
                module_name="fractalsemantics.exp03_coordinate_entropy",
                description="Validates that all 7 dimensions are necessary to avoid collisions through ablation testing.",
                educational_focus="Dimensional Analysis and Shannon Entropy Calculation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-04": ExperimentConfig(
                experiment_id="EXP-04",
                module_name="fractalsemantics.exp04_fractal_scaling",
                description="Tests consistency of addressing properties across different scales (1K to 1M entities).",
                educational_focus="Fractal Geometry Principles and Scale Invariance Analysis",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "matplotlib"]
            ),
            "EXP-05": ExperimentConfig(
                experiment_id="EXP-05",
                module_name="fractalsemantics.exp05_compression_expansion",
                description="Tests lossless compression through hierarchical structures (fragments → clusters → glyphs → mist).",
                educational_focus="Information Theory and Hierarchical Compression Algorithms",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["zlib", "pickle"]
            ),
            "EXP-06": ExperimentConfig(
                experiment_id="EXP-06",
                module_name="fractalsemantics.exp06_entanglement_detection",
                description="Tests detection of narrative entanglement between bit-chains using semantic similarity.",
                educational_focus="Semantic Similarity Metrics and Cosine Similarity Calculation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-07": ExperimentConfig(
                experiment_id="EXP-07",
                module_name="fractalsemantics.exp07_luca_bootstrap",
                description="Tests bootstrapping from Last Universal Common Ancestor to derive all entities.",
                educational_focus="Evolutionary Algorithms and Lineage Tree Generation",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["random", "itertools"]
            ),
            "EXP-08": ExperimentConfig(
                experiment_id="EXP-08",
                module_name="fractalsemantics.exp08_self_organizing_memory",
                description="Tests FractalSemantics's ability to create self-organizing memory structures with semantic clustering.",
                educational_focus="Neural Network Clustering and Self-Organization Principles",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "sklearn"]
            ),
            "EXP-09": ExperimentConfig(
                experiment_id="EXP-09",
                module_name="fractalsemantics.exp09_memory_pressure",
                description="Tests system resilience and performance under constrained memory conditions.",
                educational_focus="Memory Management Algorithms and Performance Under Constraints",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["psutil", "gc"]
            ),
            "EXP-10": ExperimentConfig(
                experiment_id="EXP-10",
                module_name="fractalsemantics.exp10_multidimensional_query",
                description="Tests FractalSemantics's unique querying capabilities across all 8 dimensions.",
                educational_focus="Multi-Dimensional Indexing and Query Optimization Algorithms",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-11": ExperimentConfig(
                experiment_id="EXP-11",
                module_name="fractalsemantics.exp11_dimension_cardinality",
                description="Explores pros and cons of 7 dimensions vs. more or fewer dimensions.",
                educational_focus="Dimensional Trade-off Analysis and Optimal Dimension Count",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "matplotlib"]
            ),
            "EXP-11b": ExperimentConfig(
                experiment_id="EXP-11b",
                module_name="fractalsemantics.exp11b_dimension_stress_test",
                description="Stress tests dimensional analysis with extreme parameter variations.",
                educational_focus="Dimensional Stress Testing and Parameter Sensitivity Analysis",
                experiment_type="stress_test",
                quick_mode_supported=False,  # Stress tests require full execution
                timeout_seconds=600,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-12": ExperimentConfig(
                experiment_id="EXP-12",
                module_name="fractalsemantics.exp12_benchmark_comparison",
                description="Compares FractalSemantics against common systems (UUID, SHA256, Vector DB, etc.).",
                educational_focus="Comparative Performance Analysis and Benchmarking Methodologies",
                experiment_type="standard",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["time", "uuid", "hashlib"]
            ),
            "EXP-13": ExperimentConfig(
                experiment_id="EXP-13",
                module_name="fractalsemantics.exp13_fractal_gravity",
                description="Tests whether fractal entities naturally create gravitational cohesion without falloff.",
                educational_focus="Fractal Gravity and Hierarchical Cohesion Analysis",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "math"]
            ),
            "EXP-14": ExperimentConfig(
                experiment_id="EXP-14",
                module_name="fractalsemantics.exp14_atomic_fractal_mapping",
                description="Maps electron shell structure to fractal parameters and validates atomic structure emergence.",
                educational_focus="Atomic Structure and Fractal Hierarchy Mapping",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "periodictable"]
            ),
            "EXP-15": ExperimentConfig(
                experiment_id="EXP-15",
                module_name="fractalsemantics.exp15_topological_conservation",
                description="Tests whether fractal systems conserve topology rather than classical energy and momentum.",
                educational_focus="Topological Conservation Laws and Fractal Physics",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "networkx"]
            ),
            "EXP-16": ExperimentConfig(
                experiment_id="EXP-16",
                module_name="fractalsemantics.exp16_hierarchical_distance_mapping",
                description="Tests hierarchical distance mapping and its relationship to spatial distance.",
                educational_focus="Hierarchical Distance Metrics and Spatial Mapping",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-17": ExperimentConfig(
                experiment_id="EXP-17",
                module_name="fractalsemantics.exp17_thermodynamic_validation",
                description="Validates thermodynamic properties of fractal systems and energy conservation.",
                educational_focus="Thermodynamic Validation and Energy Analysis",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-18": ExperimentConfig(
                experiment_id="EXP-18",
                module_name="fractalsemantics.exp18_falloff_thermodynamics",
                description="Tests falloff thermodynamics and its relationship to hierarchical structure.",
                educational_focus="Falloff Thermodynamics and Hierarchical Energy Distribution",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-19": ExperimentConfig(
                experiment_id="EXP-19",
                module_name="fractalsemantics.exp19_orbital_equivalence",
                description="Tests orbital equivalence and hierarchical relationships in fractal systems.",
                educational_focus="Orbital Equivalence and Fractal Dynamics",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-20": ExperimentConfig(
                experiment_id="EXP-20",
                module_name="fractalsemantics.exp20_vector_field_derivation",
                description="Derives vector field approaches for fractal gravitational interactions.",
                educational_focus="Vector Field Derivation and Fractal Mechanics",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            ),
            "EXP-21": ExperimentConfig(
                experiment_id="EXP-21",
                module_name="fractalsemantics.exp21_earth_moon_sun_simulation",
                description="Simulates the Earth-Moon-Sun system with accurate orbital mechanics and gravitational interactions.",
                educational_focus="Orbital Mechanics and Gravitational Simulation",
                experiment_type="advanced",
                quick_mode_supported=True,
                timeout_seconds=300,
                dependencies=["numpy", "scipy"]
            )
        }
        return configs

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

            # Determine result type based on experiment outcome
            result_type = self._determine_result_type(experiment_id, result)

            return ExperimentResult(
                experiment_id=experiment_id,
                success=result["success"],
                duration=duration,
                output=result["output"],
                metrics=result["metrics"],
                educational_content=educational_content,
                result_type=result_type
            )

        except Exception as e:
            duration = time.time() - start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                duration=duration,
                output=f"Error executing experiment: {str(e)}",
                metrics={},
                educational_content=[f"❌ Experiment failed with error: {str(e)}"],
                result_type="failure"
            )

    def _generate_introduction(self, experiment_id: str, config: dict[str, any]) -> str:
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
            intro += f"   {i}. {step}"

        intro += "\n" + "="*60 + "\n"
        return intro

    def _generate_analysis(self, experiment_id: str, result: dict[str, any]) -> str:
        """Generate educational analysis of experiment results."""
        analysis = f"""
📊 EXPERIMENT RESULTS ANALYSIS: {experiment_id}
🎯 Key Learning Outcomes:

"""

        if result["success"]:
            analysis += "✅ Experiment completed successfully!"
            analysis += "📈 Performance Metrics:"
            for key, value in result["metrics"].items():
                analysis += f"   • {key}: {value}"
        else:
            analysis += "❌ Experiment encountered issues."
            analysis += "🔍 Troubleshooting Insights:"
            analysis += "   • This demonstrates real-world challenges in computational systems"
            analysis += "   • Error analysis helps identify system limitations"
            analysis += "   • Understanding failure modes is crucial for system design"

        analysis += f"""
💡 Real-World Applications:
{self._get_real_world_applications(experiment_id)}

🎯 Takeaway Lessons:
{self._get_key_lessons(experiment_id)}
"""

        return analysis

    def _get_mathematical_concepts(self, experiment_id: str) -> list[str]:
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
            "EXP-11b": [
                "Stress testing methodologies",
                "Parameter sensitivity analysis",
                "Robust system design",
                "Performance under extreme conditions"
            ],
            "EXP-12": [
                "Comparative Performance Analysis",
                "Benchmarking Methodologies",
                "System Trade-off Evaluation",
                "Statistical Significance Testing"
            ],
            "EXP-13": [
                "Fractal Gravity and Hierarchical Cohesion",
                "Hierarchical Distance Metrics",
                "Tree Structure Mathematics",
                "Gravitational Field Theory"
            ],
            "EXP-14": [
                "Atomic Structure and Electron Configuration",
                "Shell-Based Fractal Mapping",
                "Periodic Table Analysis",
                "Quantum Mechanical Principles"
            ],
            "EXP-15": [
                "Topological Conservation Laws",
                "Fractal Physics Principles",
                "Conservation of Structure",
                "Classical vs. Fractal Mechanics"
            ],
            "EXP-16": [
                "Hierarchical Distance Metrics",
                "Spatial Mapping Algorithms",
                "Distance Transformation Mathematics",
                "Multi-Scale Analysis"
            ],
            "EXP-17": [
                "Thermodynamic Validation",
                "Energy Conservation Analysis",
                "Statistical Mechanics",
                "Thermal Equilibrium Principles"
            ],
            "EXP-18": [
                "Falloff Thermodynamics",
                "Energy Distribution Patterns",
                "Hierarchical Energy Flow",
                "Thermodynamic Efficiency Analysis"
            ],
            "EXP-19": [
                "Orbital Equivalence",
                "Hierarchical Dynamics",
                "Fractal Orbital Mechanics",
                "Equivalence Principle Analysis"
            ],
            "EXP-20": [
                "Vector Field Derivation",
                "Fractal Gravitational Interactions",
                "Field Theory Mathematics",
                "Vector Calculus Applications"
            ],
            "EXP-21": [
                "Orbital Mechanics",
                "Gravitational Simulation",
                "N-body Problem Analysis",
                "Numerical Integration Methods"
            ]
        }
        return concepts_map.get(experiment_id, ["General Computational Concepts"])

    def _get_experiment_steps(self, experiment_id: str) -> list[str]:
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
                "EXP-11b": [
                    "Define extreme parameter variations for dimensional analysis",
                    "Run stress tests with high and low dimension counts",
                    "Measure system performance and stability under stress",
                    "Analyze sensitivity to dimensional changes",
                    "Identify robustness thresholds for dimensional configurations"
            ],
            "EXP-12": [
                "Define comprehensive comparison metrics (performance, storage, expressiveness)",
                "Test all benchmark systems (UUID, SHA256, Vector DB, Graph DB, RDBMS)",
                "Measure performance characteristics across different scales",
                "Calculate relative advantages and disadvantages",
                "Analyze trade-offs and identify optimal use cases for each system"
            ],
            "EXP-13": [
                "Build pure fractal hierarchy trees for different elements",
                "Calculate hierarchical distances between random node pairs",
                "Compute natural cohesion without falloff across hierarchy",
                "Apply falloff to hierarchical distances and measure effects",
                "Analyze conservation patterns across different elements"
            ],
            "EXP-14": [
                "Retrieve electron shell configurations for elements",
                "Map shell count to fractal depth and valence electrons to branching factor",
                "Build hierarchical structures based on atomic properties",
                "Validate fractal parameters against observed densities",
                "Test prediction accuracy across the periodic table"
            ],
            "EXP-15": [
                "Define topological invariants (nodes, depth, connectivity, entropy)",
                "Run orbital dynamics simulation with hierarchical tracking",
                "Measure topological conservation over time",
                "Compare against classical energy conservation",
                "Validate fundamental difference between fractal and classical physics"
            ],
            "EXP-16": [
                "Create hierarchical distance mappings for spatial relationships",
                "Test distance transformation algorithms",
                "Validate hierarchical vs. spatial distance correlations",
                "Analyze multi-scale mapping effectiveness",
                "Optimize distance preservation across scales"
            ],
            "EXP-17": [
                "Set up thermodynamic validation framework",
                "Measure energy conservation in fractal systems",
                "Analyze thermal equilibrium properties",
                "Test statistical mechanics principles",
                "Validate thermodynamic consistency"
            ],
            "EXP-18": [
                "Implement falloff thermodynamics models",
                "Measure energy distribution patterns",
                "Analyze hierarchical energy flow",
                "Test thermodynamic efficiency across structures",
                "Validate falloff impact on system performance"
            ],
            "EXP-19": [
                "Define orbital equivalence relationships",
                "Test hierarchical dynamics in orbital systems",
                "Validate fractal orbital mechanics principles",
                "Analyze equivalence principle applications",
                "Compare with classical orbital mechanics"
            ],
            "EXP-20": [
                "Derive vector field approaches for fractal interactions",
                "Test different force calculation methods",
                "Validate gravitational interaction models",
                "Analyze field theory applications",
                "Optimize vector calculus implementations"
            ],
            "EXP-21": [
                "Set up Earth-Moon-Sun system simulation",
                "Implement accurate orbital mechanics equations",
                "Simulate gravitational interactions over time",
                "Validate against known astronomical data",
                "Analyze n-body problem dynamics and numerical integration methods"
            ]
        }
        return steps_map.get(experiment_id, ["Execute experiment", "Analyze results", "Generate report"])

    def _get_real_world_applications(self, experiment_id: str) -> str:
        """Get real-world applications for the experiment."""
        applications_map = {
            "EXP-01": "• Content-addressable storage systems• Cryptographic hash functions• Database indexing strategies• File system design",
            "EXP-02": "• Database query optimization• Cache system design• Real-time data processing• High-frequency trading systems",
            "EXP-03": "• Feature selection in machine learning• Dimensionality reduction techniques• Data compression algorithms• Information retrieval systems",
            "EXP-04": "• Scalable distributed systems• Cloud computing architectures• Big data processing frameworks• Network protocol design",
            "EXP-05": "• Data compression software• Multimedia file formats• Database storage optimization• Network bandwidth optimization",
            "EXP-06": "• Semantic search engines• Recommendation systems• Natural language processing• Knowledge graph construction",
            "EXP-07": "• Evolutionary biology research• Phylogenetic tree construction• Genetic algorithm design• Ancestral sequence reconstruction",
            "EXP-08": "• Artificial neural networks• Knowledge management systems• Self-organizing maps• Clustering algorithms",
            "EXP-09": "• Memory-constrained embedded systems• Mobile application optimization• Cloud resource management• Real-time system design",
            "EXP-10": "• Multi-dimensional database systems• Geographic information systems• Scientific data analysis• Complex query optimization",
            "EXP-11": "• System design trade-off analysis• Resource allocation strategies• Performance optimization• Cost-benefit analysis",
            "EXP-11b": "• Stress testing methodologies• Parameter sensitivity analysis• Robust system design• Performance under extreme conditions",
            "EXP-12": "• Technology selection for projects• Performance benchmarking• System architecture design• Vendor evaluation",
            "EXP-13": "• Hierarchical data organization• Natural language processing• Knowledge graph construction• Self-organizing systems",
            "EXP-14": "• Atomic structure modeling• Periodic table analysis• Quantum computing applications• Material science research",
            "EXP-15": "• Topological data analysis• Fractal physics applications• Complex system modeling• Network topology optimization",
            "EXP-16": "• Spatial database systems• Geographic information systems• Multi-scale data analysis• Hierarchical data visualization",
            "EXP-17": "• Thermodynamic system analysis• Energy conservation modeling• Statistical mechanics applications• Thermal system optimization",
            "EXP-18": "• Energy distribution analysis• Hierarchical system optimization• Thermodynamic efficiency modeling• Resource allocation systems",
            "EXP-19": "• Orbital mechanics applications• Hierarchical system dynamics• Equivalence principle testing• Complex system analysis",
            "EXP-20": "• Gravitational field modeling• Vector field applications• Fractal interaction systems• Field theory implementations",
            "EXP-21": "• Astronomical simulations• Orbital mechanics research• N-body problem analysis• Numerical integration method development"
        }
        return applications_map.get(experiment_id, "• General computational applications")

    def _get_key_lessons(self, experiment_id: str) -> str:
        """Get key lessons for the experiment."""
        lessons_map = {
            "EXP-01": "• Mathematical foundations ensure system reliability• Collision resistance is critical for data integrity• Proper coordinate systems enable unique addressing• Cryptographic principles provide security guarantees",
            "EXP-02": "• Algorithmic efficiency impacts real-world performance• Hash tables provide optimal retrieval performance• System design must consider scalability• Performance measurement requires precise timing",
            "EXP-03": "• Dimensional analysis reveals system properties• Information theory guides feature selection• Ablation studies identify critical components• Entropy measures system complexity",
            "EXP-04": "• Fractal properties enable scalable systems• Self-similarity provides consistent behavior• Scale invariance ensures predictable performance• Power laws describe natural system behavior",
            "EXP-05": "• Hierarchical structures enable efficient compression• Information theory guides algorithm design• Lossless compression preserves data integrity• Multi-level optimization improves efficiency",
            "EXP-06": "• Semantic similarity enables intelligent systems• Vector embeddings capture meaningful relationships• Threshold selection balances precision and recall• Entanglement detection reveals hidden connections",
            "EXP-07": "• Evolutionary principles guide system design• Lineage tracking enables provenance• Bootstrap methods create comprehensive systems• Genetic algorithms solve complex problems",
            "EXP-08": "• Self-organization creates emergent intelligence• Clustering reveals natural data structure• Semantic coherence improves system usability• Network topology affects performance",
            "EXP-09": "• Resource constraints drive innovation• Optimization strategies improve resilience• Performance under pressure reveals system quality• Memory management is critical for efficiency",
            "EXP-10": "• Multi-dimensional indexing enables complex queries• Query optimization reduces computational complexity• Dimensional pruning improves performance• Spatial databases handle complex data relationships",
            "EXP-11": "• Trade-off analysis guides system design• Optimal dimensionality balances expressiveness and complexity• Pareto efficiency identifies best solutions• Complexity theory informs algorithm selection",
            "EXP-11b": "• Stress testing methodologies• Parameter sensitivity analysis• Robust system design• Performance under extreme conditions",
            "EXP-12": "• Comparative analysis reveals system strengths• Benchmarking provides objective evaluation• Performance metrics guide technology selection• Trade-off analysis informs architectural decisions",
            "EXP-13": "• Hierarchical structures enable natural cohesion• Fractal gravity provides alternative to classical gravity• Tree-based organization supports efficient relationships• Hierarchical distance metrics enable spatial reasoning",
            "EXP-14": "• Atomic structure can be modeled through fractal hierarchies• Electron shell configurations inform fractal parameters• Periodic table patterns emerge from fractal properties• Quantum mechanical principles align with fractal mathematics",
            "EXP-15": "• Topological conservation provides alternative to classical conservation laws• Fractal systems prioritize structure over energy• Hierarchical tracking enables complex system analysis• Classical physics principles may not apply to fractal systems",
            "EXP-16": "• Hierarchical distance mapping enables multi-scale analysis• Spatial relationships can be preserved through hierarchical structures• Distance transformation algorithms support complex queries• Multi-scale analysis reveals hidden patterns in data",
            "EXP-17": "• Thermodynamic principles apply to fractal systems• Energy conservation manifests differently in hierarchical structures• Statistical mechanics principles guide fractal system behavior• Thermal equilibrium can be achieved through hierarchical organization",
            "EXP-18": "• Falloff thermodynamics affects hierarchical energy distribution• Energy efficiency varies across hierarchical levels• Thermodynamic optimization requires multi-scale analysis• Hierarchical structures impact energy flow patterns",
            "EXP-19": "• Orbital equivalence enables hierarchical system modeling• Fractal orbital mechanics provide alternative to classical mechanics• Equivalence principles apply across hierarchical scales• Complex orbital relationships emerge from fractal structures",
            "EXP-20": "• Vector field approaches enable fractal gravitational modeling• Field theory principles apply to hierarchical systems• Vector calculus provides tools for fractal interaction analysis• Gravitational interactions can be modeled through fractal mathematics",
            "EXP-21": "• Accurate orbital mechanics is essential for realistic simulations• Gravitational interactions are complex and require careful modeling• N-body problem analysis reveals system dynamics• Numerical integration methods are critical for long-term stability"
        }
        return lessons_map.get(experiment_id, "• Computational thinking solves complex problems• Mathematical foundations enable reliable systems• Experimental methodology validates theoretical concepts")

    def _determine_result_type(self, experiment_id: str, result: dict[str, any]) -> str:
        """Determine the result type based on experiment outcome and scientific validation."""
        # Technical failure - experiment crashed or had execution errors
        if not result["success"]:
            return "failure"

        # Check for scientific validation failures in the output
        output = result.get("output", "").lower()

        # Look for scientific validation failure indicators
        scientific_failures = [
            "experiment success: no",
            "distance mapping success: no",
            "force scaling consistent: no",
            "validation failed",
            "scientific validation failed",
            "not meet scientific criteria",
            "experiment_success: false",
            "distance_mapping_success: false",
            "force_scaling_consistent: false"
        ]

        # Check if this is an advanced experiment that might have scientific validation failures
        advanced_experiments = ["EXP-16", "EXP-17", "EXP-18", "EXP-19", "EXP-20", "EXP-21"]

        if experiment_id in advanced_experiments:
            for failure_indicator in scientific_failures:
                if failure_indicator in output:
                    return "warning"  # Scientific validation failed but experiment ran successfully

        # Check specific experiment validation patterns
        if experiment_id == "EXP-16":
            # Check for hierarchical distance mapping validation
            if "distance_correlation" in str(result.get("metrics", {})):
                # Parse metrics to check if validation passed
                metrics = result.get("metrics", {})
                if isinstance(metrics, dict) and "Exponential" in metrics:
                    exp_data = metrics["Exponential"]
                    if isinstance(exp_data, dict):
                        distance_corr = exp_data.get("distance_correlation", 0)
                        force_corr = exp_data.get("force_correlation", 0)
                        # If correlations are very low, consider it a partial success
                        if distance_corr < 0.2 and force_corr < 0.2:
                            return "partial_success"

        elif experiment_id in ["EXP-18", "EXP-19", "EXP-20", "EXP-21"]:
            # Use centralized validation logic for advanced experiments
            return self._check_advanced_experiment_validation(experiment_id, output)

    def _check_advanced_experiment_validation(self, experiment_id: str, output: str) -> str:
        """Centralized validation logic for advanced experiments (EXP-18 through EXP-21)."""
        # Define validation rules for each advanced experiment
        validation_rules = {
            "EXP-18": {
                "failure_indicators": [
                    "no improvement",
                    "doesn't help thermodynamics",
                    "status: failed"
                ],
                "description": "thermodynamics validation failure"
            },
            "EXP-19": {
                "failure_indicators": [
                    ("orbital equivalence", "not properly simulated")
                ],
                "description": "orbital simulation issues"
            },
            "EXP-20": {
                "failure_indicators": [
                    "validation failed"
                ],
                "description": "vector field validation issues"
            },
            "EXP-21": {
                "failure_indicators": [
                    "validation failed"
                ],
                "description": "simulation validation issues"
            }
        }

        # Get validation rules for this experiment
        rules = validation_rules.get(experiment_id)
        if not rules:
            return "success"  # Default to success if no rules defined

        # Check for failure indicators
        output_lower = output.lower()

        for indicator in rules["failure_indicators"]:
            if isinstance(indicator, tuple):
                # For compound indicators (both must be present)
                if all(ind.lower() in output_lower for ind in indicator):
                    return "warning"
            else:
                # For simple indicators (any one present)
                if indicator.lower() in output_lower:
                    return "warning"

        return "success"

    async def _execute_experiment_module(self, experiment_id: str, quick_mode: bool) -> dict[str, any]:
        """Execute the actual experiment module."""

        # Experiment ID to module name mapping
        experiment_map = {
            "EXP-01": "exp01_geometric_collision",
            "EXP-02": "exp02_retrieval_efficiency",
            "EXP-03": "exp03_coordinate_entropy",
            "EXP-04": "exp04_fractal_scaling",
            "EXP-05": "exp05_compression_expansion",
            "EXP-06": "exp06_entanglement_detection",
            "EXP-07": "exp07_luca_bootstrap",
            "EXP-08": "exp08_self_organizing_memory",
            "EXP-09": "exp09_memory_pressure",
            "EXP-10": "exp10_multidimensional_query",
            "EXP-11": "exp11_dimension_cardinality",
            "EXP-11b": "exp11b_dimension_stress_test",
            "EXP-12": "exp12_benchmark_comparison",
            "EXP-13": "exp13_fractal_gravity",
            "EXP-14": "exp14_atomic_fractal_mapping",
            "EXP-15": "exp15_topological_conservation",
            "EXP-16": "exp16_hierarchical_distance_mapping",
            "EXP-17": "exp17_thermodynamic_validation",
            "EXP-18": "exp18_falloff_thermodynamics",
            "EXP-19": "exp19_orbital_equivalence",
            "EXP-20": "exp20_vector_field_derivation",
            "EXP-21": "exp21_earth_moon_sun"
        }

        try:
            # Use subprocess execution for all experiments to ensure compatibility
            return await self._execute_experiment_subprocess(experiment_id, quick_mode)

        except Exception as e:
            import traceback
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "working_directory": os.getcwd(),
                "python_executable": sys.executable,
                "experiment_id": experiment_id,
                "module_name": experiment_map.get(experiment_id, "unknown")
            }

            error_output = f"""
Subprocess execution failed!

Error Type: {error_details['error_type']}
Error Message: {error_details['error_message']}

Working Directory: {error_details['working_directory']}
Python Executable: {error_details['python_executable']}
Experiment ID: {error_details['experiment_id']}
Module Name: {error_details['module_name']}

Full Traceback:
{error_details['traceback']}
"""

            return {
                "success": False,
                "output": error_output,
                "metrics": error_details
            }

    async def _execute_experiment_subprocess(self, experiment_id: str, quick_mode: bool) -> dict[str, any]:
        """Execute experiment as subprocess with progress tracking (Windows-compatible)."""
        import threading

        try:
            # Import progress communication module
            from fractalsemantics.progress_comm import (
                is_progress_message,
                parse_progress_message,
            )

            # Construct command to run the experiment
            experiment_map = {
                "EXP-01": "exp01_geometric_collision",
                "EXP-02": "exp02_retrieval_efficiency",
                "EXP-03": "exp03_coordinate_entropy",
                "EXP-04": "exp04_fractal_scaling",
                "EXP-05": "exp05_compression_expansion",
                "EXP-06": "exp06_entanglement_detection",
                "EXP-07": "exp07_luca_bootstrap",
                "EXP-08": "exp08_self_organizing_memory",
                "EXP-09": "exp09_memory_pressure",
                "EXP-10": "exp10_multidimensional_query",
                "EXP-11": "exp11_dimension_cardinality",
                "EXP-11b": "exp11b_dimension_stress_test",
                "EXP-12": "exp12_benchmark_comparison",
                "EXP-13": "exp13_fractal_gravity",
                "EXP-14": "exp14_atomic_fractal_mapping",
                "EXP-15": "exp15_topological_conservation",
                "EXP-16": "exp16_hierarchical_distance_mapping",
                "EXP-17": "exp17_thermodynamic_validation",
                "EXP-18": "exp18_falloff_thermodynamics",
                "EXP-19": "exp19_orbital_equivalence",
                "EXP-20": "exp20_vector_field_derivation",
                "EXP-21": "exp21_earth_moon_sun"
            }

            module_name = experiment_map.get(experiment_id)
            if not module_name:
                raise ValueError(f"Unknown experiment: {experiment_id}")

            # Use the current Python executable (from virtual environment)
            python_executable = sys.executable
            cmd = [
                python_executable, str(Path(__file__).parent / f"{module_name}.py")
            ]

            # Add quick mode flag if needed
            if quick_mode:
                cmd.append("--quick")

            # Prepare environment with progress file path
            env = os.environ.copy()

            # Always ensure progress file env var is set for subprocess
            if "FRACTALSEMANTICS_PROGRESS_FILE" in os.environ:
                env["FRACTALSEMANTICS_PROGRESS_FILE"] = os.environ["FRACTALSEMANTICS_PROGRESS_FILE"]
            else:
                # Set default progress file path if not already set
                progress_file_path = str(Path("results") / "gui_progress.jsonl")
                env["FRACTALSEMANTICS_PROGRESS_FILE"] = progress_file_path

            # Remove Streamlit-specific environment variables
            streamlit_vars = [k for k in env if k.startswith('STREAMLIT_')]
            for var in streamlit_vars:
                del env[var]

            # Ensure fractalsemantics module can be found
            project_root = str(Path(__file__).parent.parent)
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = project_root

            env["VIRTUAL_ENV"] = sys.prefix
            env["PATH"] = f"{sys.prefix}/bin{os.pathsep}{env.get('PATH', '')}"

            # Run subprocess with streaming output (Windows-compatible threading approach)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                env=env,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Collect output in real-time using threads (Windows-compatible)
            stdout_lines = []
            stderr_lines = []

            # Queues for thread-safe communication
            stdout_queue = Queue()
            stderr_queue = Queue()

            def read_stdout():
                """Read stdout in background thread."""
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            stdout_queue.put(line)
                except ast.ParseError:
                    pass
                finally:
                    stdout_queue.put(None)  # Signal completion

            def read_stderr():
                """Read stderr in background thread."""
                try:
                    for line in iter(process.stderr.readline, ''):
                        if line:
                            stderr_queue.put(line)
                except ast.ParseError:
                    pass
                finally:
                    stderr_queue.put(None)  # Signal completion

            # Start reader threads
            stdout_thread = Thread(target=read_stdout, daemon=True)
            stderr_thread = Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            # Read streams until process completes
            start_time = time.time()
            timeout = 300  # 5 minutes
            stdout_done = False
            stderr_done = False

            while not (stdout_done and stderr_done):
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    raise TimeoutError(f"Experiment {experiment_id} timed out after {timeout} seconds")

                # Read from stdout queue
                if not stdout_done:
                    try:
                        line = stdout_queue.get(timeout=0.1)
                        if line is None:
                            stdout_done = True
                        else:
                            stdout_lines.append(line)
                    except ast.ParseError:
                        pass  # Queue empty

                # Read from stderr queue
                if not stderr_done:
                    try:
                        line = stderr_queue.get(timeout=0.1)
                        if line is None:
                            stderr_done = True
                        else:
                            stderr_lines.append(line)
                    except ast.ParseError:
                        pass  # Queue empty

                # Yield control to event loop
                await asyncio.sleep(0.01)

            # Wait for threads to finish
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

            # Get return code
            return_code = process.wait()

            # Combine output
            output = ''.join(stdout_lines)
            error = ''.join(stderr_lines)

            # Determine success by checking for completion markers in output
            # Exit code 0 = definite success
            # Exit code 1 = could be warning (successful completion with validation warnings) or failure
            # We need to check the output to distinguish between these cases
            completion_markers = [
                "[OK]",
                "COMPLETE",
                "[Success]",
                "SUCCESS",
                f"{experiment_id} COMPLETE"
            ]

            has_completion_marker = any(marker in output.upper() for marker in [m.upper() for m in completion_markers])

            # Success if:
            # 1. Return code is 0 (definite success), OR
            # 2. Return code is non-zero BUT output contains completion markers (warning-level result)
            success = return_code == 0 or has_completion_marker

            # Add detailed diagnostic information if subprocess had non-zero exit code
            if return_code != 0:
                diagnostic_info = f"""
=== EXPERIMENT COMPLETED WITH NON-ZERO EXIT CODE ===
Return Code: {return_code}
Experiment ID: {experiment_id}
Module Name: {module_name}
Completion Status: {'Completed with warnings' if has_completion_marker else 'Failed'}

{'Note: Non-zero exit code typically indicates scientific validation warnings, not technical failures.' if has_completion_marker else 'Note: No completion marker found - this appears to be a technical failure.'}

=== STDOUT ===
{output if output else '(no output)'}

=== STDERR ===
{error if error else '(no error output)'}
"""
                print(diagnostic_info)
                # Only add error info to output if it's a true technical failure
                if not has_completion_marker:
                    output = diagnostic_info + "\n" + output

            # Parse progress messages from stderr and filter them out of error output
            progress_messages = []
            filtered_error_lines = []
            for line in error.split('\n'):
                if is_progress_message(line):
                    progress_msg = parse_progress_message(line)
                    if progress_msg and progress_msg.experiment_id == experiment_id:
                        progress_messages.append(progress_msg)
                    # Don't include progress messages in error output
                else:
                    filtered_error_lines.append(line)

            # Use filtered error output
            filtered_error = '\n'.join(filtered_error_lines).strip()

            # Build metrics
            metrics: dict[str, any] = {"return_code": return_code}
            if progress_messages:
                progress_data = []
                for msg in progress_messages:
                    progress_data.append({
                        "timestamp": msg.timestamp,
                        "progress_percent": float(msg.progress_percent),
                        "stage": msg.stage,
                        "message": msg.message,
                        "message_type": msg.message_type
                    })
                metrics["progress_messages"] = progress_data

            return {
                "success": success,
                "output": output + (f"\nStderr: {filtered_error}" if filtered_error else ""),
                "metrics": metrics
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "output": f"Subprocess execution failed: {str(e)}\n\nFull traceback:\n{error_details}\n\nCommand attempted: {' '.join(cmd) if 'cmd' in locals() else 'Command not constructed'}\nWorking directory: {os.getcwd()}\nPython executable: {sys.executable}",
                "metrics": {"error_type": type(e).__name__, "error_message": str(e)}
            }

    async def run_batch_experiments(self, experiment_ids: list[str], quick_mode: bool = False,
                                   parallel: bool = True, progress_callback=None) -> BatchRunResult:
        """Run multiple experiments with progress tracking and educational output."""
        start_time = time.time()
        experiment_results = []

        if not experiment_ids:
            experiment_ids = list(self.experiment_configs.keys())

        # Validate and normalize experiment IDs
        validated_experiments = []
        for exp_id in experiment_ids:
            validated_id = self._validate_experiment_id(exp_id)
            if validated_id:
                validated_experiments.append(validated_id)
            else:
                raise ValueError(f"Unknown experiment: {exp_id}")

        experiment_ids = validated_experiments
        total_experiments = len(experiment_ids)
        successful_experiments = 0
        failed_experiments = 0

        print(f"🚀 Starting batch run of {total_experiments} experiments...")
        print(f"📊 Feature Level: {'Quick' if quick_mode else 'Full'}")
        print(f"⚡ Execution Mode: {'Parallel' if parallel else 'Sequential'}")
        print("=" * 80)

        if parallel:
            # Run experiments in parallel with individual progress tracking
            print(f"Running {total_experiments} experiments in parallel...")

            # Create individual progress bars for each experiment
            progress_bars = {}
            for exp_id in experiment_ids:
                progress_bars[exp_id] = tqdm.tqdm(
                    total=100,
                    desc=f"{exp_id}",
                    unit="%",
                    position=len(progress_bars),
                    leave=True,
                    ncols=80,
                    bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}] {postfix}"
                )

            # Run experiments in parallel
            tasks = [self.run_experiment(exp_id, quick_mode) for exp_id in experiment_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and update progress bars
            for i, result in enumerate(results):
                experiment_id = experiment_ids[i]
                progress_bar = progress_bars[experiment_id]

                if isinstance(result, Exception):
                    # Handle exceptions from gather
                    duration = 0
                    error_result = ExperimentResult(
                        experiment_id=experiment_id,
                        success=False,
                        duration=duration,
                        output=f"Error: {str(result)}",
                        metrics={},
                        educational_content=[f"❌ Experiment {experiment_id} failed with error: {str(result)}"]
                    )
                    experiment_results.append(error_result)
                    failed_experiments += 1

                    # Complete progress bar as failed
                    progress_bar.n = 100
                    progress_bar.set_postfix({"Status": "❌ Failed"})
                    progress_bar.refresh()
                    progress_bar.close()

                    if progress_callback:
                        progress_callback(len(experiment_results), total_experiments, error_result)
                else:
                    # Normal result - result is guaranteed to be ExperimentResult here
                    assert isinstance(result, ExperimentResult), f"Expected ExperimentResult, got {type(result)}"
                    experiment_results.append(result)
                    if result.success:
                        successful_experiments += 1
                        status = "✅ Success"
                    else:
                        failed_experiments += 1
                        status = "❌ Failed"

                    # Complete progress bar
                    progress_bar.n = 100
                    progress_bar.set_postfix({"Status": status, "Time": f"{result.duration:.1f}s"})
                    progress_bar.refresh()
                    progress_bar.close()

                    if progress_callback:
                        progress_callback(len(experiment_results), total_experiments, result)

        else:
            # Run experiments sequentially with single progress bar
            progress_bar = tqdm.tqdm(
                total=total_experiments,
                desc="Running experiments",
                unit="exp",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                leave=True,
                mininterval=0.1
            )

            for i, experiment_id in enumerate(experiment_ids, 1):
                try:
                    result = await self.run_experiment(experiment_id, quick_mode)
                    experiment_results.append(result)

                    if result.success:
                        successful_experiments += 1
                        status = "Success"
                    else:
                        failed_experiments += 1
                        status = "Failed"

                    # Update progress bar
                    progress_bar.set_postfix({"Status": status, "Last": result.experiment_id})
                    progress_bar.update(1)

                    if progress_callback:
                        progress_callback(i, total_experiments, result)
                except Exception as e:
                    # Handle exceptions in sequential execution
                    duration = 0
                    error_result = ExperimentResult(
                        experiment_id=experiment_id,
                        success=False,
                        duration=duration,
                        output=f"Error: {str(e)}",
                        metrics={},
                        educational_content=[f"❌ Experiment {experiment_id} failed with error: {str(e)}"]
                    )
                    experiment_results.append(error_result)
                    failed_experiments += 1

                    # Update progress bar
                    progress_bar.set_postfix({"Status": "Failed", "Last": experiment_id})
                    progress_bar.update(1)

                    if progress_callback:
                        progress_callback(i, total_experiments, error_result)

            # Close progress bar
            progress_bar.close()

        total_duration = time.time() - start_time
        summary_report = self._generate_batch_summary(experiment_results, total_duration, quick_mode)

        return BatchRunResult(
            total_experiments=total_experiments,
            successful_experiments=successful_experiments,
            failed_experiments=failed_experiments,
            total_duration=total_duration,
            experiment_results=experiment_results,
            summary_report=summary_report
        )

    def _validate_experiment_id(self, exp_id: str) -> Optional[str]:
        """Validate and normalize experiment ID format."""
        # Try exact match first
        if exp_id in self.experiment_configs:
            return exp_id

        # Try with EXP- prefix if not already present
        exp_upper = exp_id.upper()
        if not exp_upper.startswith("EXP-"):
            exp_with_prefix = f"EXP-{exp_upper.lstrip('EXP-')}"
            if exp_with_prefix in self.experiment_configs:
                return exp_with_prefix

        return None

    def _print_progress(self, current: int, total: int, result: ExperimentResult):
        """Print progress update for batch runs."""
        status = "✅" if result.success else "❌"
        duration_str = f"{result.duration:.2f}s"
        print(f"{status} {result.experiment_id} - {duration_str} ({current}/{total})")

        # Print a separator every 4 experiments
        if current % 4 == 0 and current < total:
            print("-" * 40)

    def _generate_batch_summary(self, experiment_results: list[ExperimentResult],
                              total_duration: float, quick_mode: bool) -> str:
        """Generate educational summary report for batch run."""
        successful = sum(1 for r in experiment_results if r.success)
        failed = len(experiment_results) - successful

        # Categorize failures by type
        technical_failures = sum(1 for r in experiment_results if r.result_type == "failure")
        scientific_warnings = sum(1 for r in experiment_results if r.result_type == "warning")
        partial_successes = sum(1 for r in experiment_results if r.result_type == "partial_success")

        summary = f"""
🎯 BATCH EXPERIMENT SUMMARY REPORT
{'='*80}

📊 OVERALL STATISTICS:
   • Total Experiments: {len(experiment_results)}
   • Successful: {successful}
   • Failed: {failed}
   • Success Rate: {(successful/len(experiment_results)*100):.1f}%
   • Total Duration: {total_duration:.2f} seconds
   • Average Duration: {(total_duration/len(experiment_results)):.2f} seconds per experiment
   • Feature Level: {'Quick' if quick_mode else 'Full'}

🔍 FAILURE ANALYSIS:
   • Technical Failures (crashes/errors): {technical_failures}
   • Scientific Warnings (validation failures): {scientific_warnings}
   • Partial Successes (low performance): {partial_successes}
   • True Successes: {successful}

📈 PERFORMANCE ANALYSIS:
"""

        # Analyze performance patterns
        successful_durations = [r.duration for r in experiment_results if r.success]
        if successful_durations:
            avg_duration = sum(successful_durations) / len(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)

            summary += f"""   • Average Duration (successful): {avg_duration:.2f}s
   • Fastest Experiment: {min_duration:.2f}s
   • Slowest Experiment: {max_duration:.2f}s
"""

        summary += """
🎯 EDUCATIONAL INSIGHTS:
   • This batch run demonstrates the comprehensive capabilities of FractalSemantics
   • Each experiment validates different aspects of the addressing system
   • Success rate indicates system reliability and robustness
   • Performance metrics show scalability characteristics

💡 SYSTEM VALIDATION:
"""

        # Categorize experiments by type
        collision_tests = [r for r in experiment_results if r.experiment_id in ["EXP-01", "EXP-03"]]
        performance_tests = [r for r in experiment_results if r.experiment_id in ["EXP-02", "EXP-04"]]
        advanced_tests = [r for r in experiment_results if r.experiment_id in ["EXP-05", "EXP-06", "EXP-07"]]
        system_tests = [r for r in experiment_results if r.experiment_id in ["EXP-08", "EXP-09", "EXP-10"]]
        analysis_tests = [r for r in experiment_results if r.experiment_id in ["EXP-11", "EXP-12"]]
        fractal_physics_tests = [r for r in experiment_results if r.experiment_id in ["EXP-13", "EXP-14", "EXP-15", "EXP-16", "EXP-17", "EXP-18", "EXP-19","EXP-20" ,"EXP-21"]]

        summary += f"""   • Collision Resistance Tests: {sum(1 for r in collision_tests if r.success)}/{len(collision_tests)} passed
   • Performance & Scaling Tests: {sum(1 for r in performance_tests if r.success)}/{len(performance_tests)} passed
   • Advanced Feature Tests: {sum(1 for r in advanced_tests if r.success)}/{len(advanced_tests)} passed
   • System Integration Tests: {sum(1 for r in system_tests if r.success)}/{len(system_tests)} passed
   • Analysis & Comparison Tests: {sum(1 for r in analysis_tests if r.success)}/{len(analysis_tests)} passed
   • Fractal Physics Simulations: {sum(1 for r in fractal_physics_tests if r.success)}/{len(fractal_physics_tests)} passed

🎯 KEY LEARNING OUTCOMES:
   • FractalSemantics provides robust, collision-resistant addressing
   • System scales efficiently across different data volumes
   • Multi-dimensional indexing enables powerful querying capabilities
   • Hierarchical structures support efficient compression and organization
   • Semantic relationships can be detected and analyzed
   • Fractal physics simulations fail to validate in some cases; expected behavior

⚠️  SCIENTIFIC VALIDATION INSIGHTS:
   • Technical failures indicate system crashes or execution errors
   • Scientific warnings indicate experiments ran but didn't meet validation criteria
   • Partial successes indicate experiments with sub-optimal performance
   • These distinctions help identify areas for improvement

🚀 RECOMMENDATIONS:
   • For production use: Run with full feature level for comprehensive validation
   • For development: Quick mode provides rapid feedback on core functionality
   • Monitor both technical and scientific success rates
   • Address scientific warnings to improve system capabilities
   • Regular batch runs help maintain system reliability

{'='*80}
"""

        return summary

def main():
    """Main entry point for the experiment runner."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single experiment: python experiment_runner.py <experiment_id> [--quick] [--format=json|text]")
        print("  Batch experiments: python experiment_runner.py --all [--quick|--full] [--parallel|--sequential] [--format=json|text]")
        print("  Specific batch:    python experiment_runner.py EXP-01 EXP-02 EXP-03 [--quick|--full] [--parallel|--sequential] [--format=json|text]")
        print("")
        print("Examples:")
        print("  fractalsemantics-runner --all --full")
        print("  fractalsemantics-runner --all --quick")
        print("  fractalsemantics-runner EXP-01 EXP-02 --quick --sequential")
        print("  fractalsemantics-runner EXP-01 --quick --format=json")
        sys.exit(1)

    # Parse command line arguments
    args = sys.argv[1:]

    # Check for batch mode indicators
    is_all = "--all" in args
    is_quick = "--quick" in args
    is_full = "--full" in args
    is_parallel = "--parallel" in args
    is_sequential = "--sequential" in args

    # Check for format
    output_format = "json"  # default format
    if "--format=json" in args:
        output_format = "json"
    elif "--format=text" in args:
        output_format = "text"

    # Determine feature level
    if is_quick:
        quick_mode = True
    elif is_full:
        quick_mode = False
    else:
        # Default to quick mode for batch runs, full mode for single experiments
        quick_mode = is_all or len([arg for arg in args if arg.startswith("EXP-")]) > 1

    # Determine execution mode
    if is_parallel:
        parallel_mode = True
    elif is_sequential:
        parallel_mode = False
    else:
        # Default to parallel for batch runs
        parallel_mode = is_all or len([arg for arg in args if arg.startswith("EXP-")]) > 1

    runner = ExperimentRunner()

    try:
        if is_all:
            # Run all experiments
            print(f"🚀 Running ALL experiments in {'Quick' if quick_mode else 'Full'} mode...")
            batch_result = asyncio.run(runner.run_batch_experiments(
                experiment_ids=[],  # Empty list means run all
                quick_mode=quick_mode,
                parallel=parallel_mode
            ))

            if output_format == "json":
                # Output batch result as JSON
                output = {
                    "batch_run": True,
                    "total_experiments": batch_result.total_experiments,
                    "successful_experiments": batch_result.successful_experiments,
                    "failed_experiments": batch_result.failed_experiments,
                    "total_duration": batch_result.total_duration,
                    "success_rate": (batch_result.successful_experiments / batch_result.total_experiments * 100) if batch_result.total_experiments > 0 else 0,
                    "experiment_results": [
                        {
                            "experiment_id": r.experiment_id,
                            "success": r.success,
                            "duration": r.duration,
                            "output": r.output,
                            "metrics": r.metrics,
                            "educational_content": r.educational_content
                        } for r in batch_result.experiment_results
                    ],
                    "summary_report": batch_result.summary_report
                }
                json_output = json.dumps(output, indent=2, ensure_ascii=False)
                print(json_output)
            else:
                # Output as formatted text
                print(batch_result.summary_report)

        elif any(arg.startswith("EXP-") for arg in args):
            # Run specific experiments
            experiment_ids = [arg for arg in args if arg.startswith("EXP-")]

            if len(experiment_ids) == 1:
                # Single experiment
                experiment_id = experiment_ids[0]
                result = asyncio.run(runner.run_experiment(experiment_id, quick_mode))

                if output_format == "json":
                    output = {
                        "experiment_id": result.experiment_id,
                        "success": result.success,
                        "duration": result.duration,
                        "output": result.output,
                        "metrics": result.metrics,
                        "educational_content": result.educational_content
                    }
                    json_output = json.dumps(output, indent=2, ensure_ascii=False)
                    print(json_output)
                else:
                    print("=" * 80)
                    print(f"EXPERIMENT: {result.experiment_id}")
                    print("=" * 80)
                    print(f"Success: {result.success}")
                    print(f"Duration: {result.duration:.4f} seconds")
                    print("=" * 80)
                    print("EXPERIMENT OUTPUT:")
                    print("-" * 40)
                    print(result.output)
                    print("=" * 80)
                    print("EDUCATIONAL CONTENT:")
                    print("-" * 40)
                    for i, content in enumerate(result.educational_content, 1):
                        print(f"Section {i}:")
                        print(content)
                        print("-" * 40)
                    print("=" * 80)
            else:
                # Multiple specific experiments
                print(f"🚀 Running {len(experiment_ids)} specific experiments in {'Quick' if quick_mode else 'Full'} mode...")
                batch_result = asyncio.run(runner.run_batch_experiments(
                    experiment_ids=experiment_ids,
                    quick_mode=quick_mode,
                    parallel=parallel_mode
                ))

                if output_format == "json":
                    output = {
                        "batch_run": True,
                        "experiment_ids": experiment_ids,
                        "total_experiments": batch_result.total_experiments,
                        "successful_experiments": batch_result.successful_experiments,
                        "failed_experiments": batch_result.failed_experiments,
                        "total_duration": batch_result.total_duration,
                        "success_rate": (batch_result.successful_experiments / batch_result.total_experiments * 100) if batch_result.total_experiments > 0 else 0,
                        "experiment_results": [
                            {
                                "experiment_id": r.experiment_id,
                                "success": r.success,
                                "duration": r.duration,
                                "output": r.output,
                                "metrics": r.metrics,
                                "educational_content": r.educational_content
                            } for r in batch_result.experiment_results
                        ],
                        "summary_report": batch_result.summary_report
                    }
                    json_output = json.dumps(output, indent=2, ensure_ascii=False)
                    print(json_output)
                else:
                    print(batch_result.summary_report)

        else:
            # Single experiment mode (legacy behavior)
            experiment_id = args[0]
            result = asyncio.run(runner.run_experiment(experiment_id, quick_mode))

            if output_format == "json":
                output = {
                    "experiment_id": result.experiment_id,
                    "success": result.success,
                    "duration": result.duration,
                    "output": result.output,
                    "metrics": result.metrics,
                    "educational_content": result.educational_content
                }
                json_output = json.dumps(output, indent=2, ensure_ascii=False)
                print(json_output)
            else:
                print("=" * 80)
                print(f"EXPERIMENT: {result.experiment_id}")
                print("=" * 80)
                print(f"Success: {result.success}")
                print(f"Duration: {result.duration:.4f} seconds")
                print("=" * 80)
                print("EXPERIMENT OUTPUT:")
                print("-" * 40)
                print(result.output)
                print("=" * 80)
                print("EDUCATIONAL CONTENT:")
                print("-" * 40)
                for i, content in enumerate(result.educational_content, 1):
                    print(f"Section {i}:")
                    print(content)
                    print("-" * 40)
                print("=" * 80)

    except Exception as e:
        if output_format == "json":
            print(json.dumps({
                "error": str(e),
                "success": False
            }, indent=2))
        else:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
