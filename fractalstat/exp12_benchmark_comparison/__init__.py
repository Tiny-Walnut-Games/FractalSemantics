"""
EXP-12: Benchmark Comparison Module

This module provides comprehensive benchmark comparison capabilities for evaluating
FractalStat against established addressing and indexing systems including
UUID/GUID, SHA-256 content addressing, vector databases, graph databases,
and traditional RDBMS systems.

Main Components:
- BenchmarkComparisonExperiment: Main experiment runner
- SystemBenchmarkResult: Results for individual system benchmarks
- BenchmarkComparisonResult: Complete comparative analysis results
- BenchmarkSystem: Base class for all benchmark systems
- UUIDSystem: UUID/GUID system implementation
- SHA256System: SHA-256 content addressing implementation
- VectorDBSystem: Vector database system implementation
- GraphDBSystem: Graph database system implementation
- RDBMSSystem: Traditional RDBMS implementation
- FractalStatSystem: FractalStat 7-dimensional addressing system

Usage:
    from fractalstat.exp12_benchmark_comparison import BenchmarkComparisonExperiment
    
    experiment = BenchmarkComparisonExperiment(
        sample_size=100000,
        benchmark_systems=["uuid", "sha256", "fractalstat"],
        scales=[10000, 100000],
        num_queries=1000
    )
    results, success = experiment.run()
"""

__version__ = "1.0.0"
__author__ = "FractalSemantics Team"
__description__ = "Benchmark comparison experiment for FractalStat addressing system"

# Import main classes for public API
from .entities import (
    SystemBenchmarkResult,
    BenchmarkComparisonResult,
    BenchmarkSystem,
    UUIDSystem,
    SHA256System,
    VectorDBSystem,
    GraphDBSystem,
    RDBMSSystem,
    FractalStatSystem,
)
from .experiment import BenchmarkComparisonExperiment

# Define what gets imported with "from exp12_benchmark_comparison import *"
__all__ = [
    # Main experiment class
    "BenchmarkComparisonExperiment",
    
    # Result classes
    "SystemBenchmarkResult",
    "BenchmarkComparisonResult",
    
    # System classes
    "BenchmarkSystem",
    "UUIDSystem",
    "SHA256System",
    "VectorDBSystem",
    "GraphDBSystem",
    "RDBMSSystem",
    "FractalStatSystem",
]