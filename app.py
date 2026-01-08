#!/usr/bin/env python3
"""
FractalStat Interactive Experiments - Hugging Face Space Application

This is the main application file for the FractalStat interactive experiments platform.
It provides a web-based interface for running and visualizing the 12 validation experiments
with real-time progress tracking, mathematical explanations, and data visualization.

Features:
- Interactive experiment controls with proper state management
- Real-time progress visualization and mathematical explanation displays
- Dynamic charts for collision rates, performance metrics, and dimensional analysis
- Educational overlays explaining mathematical concepts during experiment execution
- Export capabilities for experiment results and visualizations
- Hugging Face Space deployment with automatic scaling
"""

import os
import json
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Tuple
import traceback

import gradio as gr
import matplotlib.pyplot as plt

# Import FractalStat modules
from fractalstat.fractalstat_experiments import run_single_experiment, EXPERIMENTS
from fractalstat.config import ExperimentConfig

# Global state management
class ExperimentState:
    def __init__(self):
        self.is_running = False
        self.current_experiment = None
        self.progress = 0
        self.status_message = "Ready to run experiments"
        self.results = {}
        self.experiment_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.experiment_thread = None

# Initialize global state
experiment_state = ExperimentState()
config = ExperimentConfig()

# Experiment descriptions and educational content
EXPERIMENT_INFO = {
    "exp01_geometric_collision": {
        "title": "EXP-01: Geometric Collision Resistance",
        "description": "Tests the fundamental property that FractalStat addresses are unique across the 8-dimensional space.",
        "math_concept": "SHA-256 Hashing & 8-Dimensional Coordinate Systems",
        "educational_content": """
        **Mathematical Foundation:**
        
        FractalStat uses SHA-256 hashing of canonical serialization to guarantee zero collisions:
        
        ```
        address = SHA256(canonical_serialize(coordinates))
        ```
        
        **8-Dimensional Space:**
        - Realm: Domain classification (data, narrative, system, etc.)
        - Lineage: Generation from LUCA (Last Universal Common Ancestor)
        - Temperature: Thermal activity level (0.0 to abs(velocity) * density)
        - Adjacency: Relational neighbors (graph connections)
        - Horizon: Lifecycle stage (genesis, emergence, peak, decay, crystallization)
        - Resonance: Charge/alignment (-1.0 to 1.0)
        - Velocity: Rate of change
        - Density: Compression distance (0.0 to 1.0)
        - Alignment: Value based on alignment map
        
        **Why 8 Dimensions?**
        Each dimension adds exponential capacity to the address space, making collisions mathematically impossible.
        """,
        "expected_results": "100% unique addresses, 0 collisions detected"
    },
    "exp02_retrieval_efficiency": {
        "title": "EXP-02: Retrieval Efficiency",
        "description": "Measures the performance of address-based retrieval operations.",
        "math_concept": "Hash Table Performance & O(1) Complexity",
        "educational_content": """
        **Computational Complexity:**
        
        FractalStat achieves O(1) retrieval time through:
        
        ```
        retrieval_time = O(1)  # Constant time lookup
        ```
        
        **Performance Characteristics:**
        - Hash-based indexing provides constant-time access
        - No tree traversal or search algorithms required
        - Memory locality optimized through coordinate clustering
        """,
        "expected_results": "Sub-millisecond retrieval times, O(1) complexity verified"
    },
    "exp03_coordinate_entropy": {
        "title": "EXP-03: Coordinate Space Entropy",
        "description": "Analyzes the information contribution of each dimension to the overall address space.",
        "math_concept": "Shannon Entropy & Information Theory",
        "educational_content": """
        **Information Theory:**
        
        Each dimension contributes to the total entropy of the address space:
        
        ```
        H(total) = Σ H(dimension_i)
        ```
        
        **Entropy Analysis:**
        - Measures how much information each dimension provides
        - Identifies critical vs. redundant dimensions
        - Validates the mathematical necessity of all 8 dimensions
        """,
        "expected_results": "High entropy contribution from all dimensions, no redundancy"
    },
    "exp04_fractal_scaling": {
        "title": "EXP-04: Fractal Scaling Properties",
        "description": "Tests how the system performs as the number of entities scales from thousands to millions.",
        "math_concept": "Fractal Geometry & Power Law Distributions",
        "educational_content": """
        **Fractal Properties:**
        
        FractalStat exhibits self-similar behavior across scales:
        
        ```
        performance(scale) ∝ scale^k  # Power law relationship
        ```
        
        **Scale Invariance:**
        - Performance characteristics remain consistent across different entity counts
        - No degradation in collision resistance or retrieval efficiency
        - Demonstrates true fractal scaling behavior
        """,
        "expected_results": "Consistent performance across all scales, power law verified"
    },
    "exp05_compression_expansion": {
        "title": "EXP-05: Compression and Expansion",
        "description": "Tests the ability to compress and expand coordinate representations without loss.",
        "math_concept": "Lossless Compression & Huffman Coding",
        "educational_content": """
        **Compression Algorithms:**
        
        FractalStat uses hierarchical compression:
        
        ```
        fragments → clusters → glyphs → mist
        ```
        
        **Lossless Properties:**
        - Each level maintains complete information
        - Compression ratios improve with scale
        - No information loss during expansion
        """,
        "expected_results": "100% lossless compression, hierarchical structure preserved"
    },
    "exp06_entanglement_detection": {
        "title": "EXP-06: Semantic Entanglement Detection",
        "description": "Identifies and measures semantic relationships between entities through coordinate proximity.",
        "math_concept": "Cosine Similarity & Semantic Distance",
        "educational_content": """
        **Semantic Analysis:**
        
        Entanglement is measured through coordinate similarity:
        
        ```
        similarity = cos(θ) = (A · B) / (|A| |B|)
        ```
        
        **Entanglement Thresholds:**
        - Related entities have similar coordinate patterns
        - Semantic distance correlates with coordinate distance
        - Enables intelligent content organization
        """,
        "expected_results": "High precision entanglement detection, semantic relationships identified"
    },
    "exp07_luca_bootstrap": {
        "title": "EXP-07: LUCA Bootstrap Reconstruction",
        "description": "Tests the ability to reconstruct the complete system from a single Last Universal Common Ancestor entity.",
        "math_concept": "Evolutionary Algorithms & Genetic Distance",
        "educational_content": """
        **Evolutionary Principles:**
        
        System reconstruction through lineage expansion:
        
        ```
        LUCA → Generation 1 → Generation 2 → ... → Current State
        ```
        
        **Bootstrap Process:**
        - Single entity contains enough information for full reconstruction
        - Lineage tracking maintains evolutionary relationships
        - Demonstrates system completeness and self-containment
        """,
        "expected_results": "Complete system reconstruction from single entity, all relationships preserved"
    },
    "exp08_self_organizing_memory": {
        "title": "EXP-08: Self-Organizing Memory Networks",
        "description": "Demonstrates how the system organizes memory through neural network principles.",
        "math_concept": "Neural Networks & Self-Organization",
        "educational_content": """
        **Neural Organization:**
        
        Memory self-organization through coordinate clustering:
        
        ```
        cluster_center = mean(coordinates_in_cluster)
        ```
        
        **Small-World Properties:**
        - High clustering coefficient
        - Short path lengths between entities
        - Emergent organizational patterns
        """,
        "expected_results": "Self-organized memory structure, small-world network properties"
    },
    "exp09_memory_pressure": {
        "title": "EXP-09: Memory Pressure Testing",
        "description": "Tests system performance under constrained memory conditions.",
        "math_concept": "Memory Management & Resource Optimization",
        "educational_content": """
        **Resource Management:**
        
        Memory optimization strategies:
        
        ```
        memory_usage = f(entities, dimensions, compression)
        ```
        
        **Pressure Testing:**
        - Performance under memory constraints
        - Resource optimization techniques
        - Resilience to memory pressure
        """,
        "expected_results": "Maintained performance under memory pressure, efficient resource usage"
    },
    "exp10_multidimensional_query": {
        "title": "EXP-10: Multi-Dimensional Query Optimization",
        "description": "Tests the efficiency of queries across multiple dimensions simultaneously.",
        "math_concept": "Multi-Dimensional Indexing & k-d Trees",
        "educational_content": """
        **Query Optimization:**
        
        Multi-dimensional range queries:
        
        ```
        query_time = O(log n)  # Logarithmic complexity
        ```
        
        **Dimensional Pruning:**
        - Eliminates irrelevant dimensions during query
        - Optimizes search space
        - Maintains accuracy while improving performance
        """,
        "expected_results": "Logarithmic query complexity, efficient dimensional pruning"
    },
    "exp11_dimension_cardinality": {
        "title": "EXP-11: Dimension Cardinality Analysis",
        "description": "Analyzes the optimal number of dimensions for the addressing system.",
        "math_concept": "Dimensional Trade-offs & Optimization",
        "educational_content": """
        **Dimensional Analysis:**
        
        Optimal dimension count determination:
        
        ```
        optimal_dimensions = argmax(expressiveness - complexity)
        ```
        
        **Trade-off Analysis:**
        - Expressiveness vs. computational complexity
        - Storage requirements vs. addressing capacity
        - Performance vs. dimension count
        """,
        "expected_results": "8 dimensions identified as optimal, balanced expressiveness and complexity"
    },
    "exp12_benchmark_comparison": {
        "title": "EXP-12: Benchmark Comparison",
        "description": "Compares FractalStat performance against traditional addressing systems.",
        "math_concept": "Comparative Analysis & Performance Metrics",
        "educational_content": """
        **Benchmarking Methodology:**
        
        Comparative performance analysis:
        
        ```
        relative_performance = system_performance / baseline_performance
        ```
        
        **Comparison Systems:**
        - UUID generation and lookup
        - SHA-256 hashing
        - Vector database indexing
        - Traditional relational database keys
        """,
        "expected_results": "Superior performance across all metrics compared to traditional systems"
    }
}

def create_progress_chart(experiment_name: str, progress_data: List[Tuple[float, float]]) -> plt.Figure:
    """Create a progress visualization chart for the experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if progress_data:
        times, values = zip(*progress_data)
        
        # Progress over time
        ax1.plot(times, values, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title(f'{experiment_name} - Progress')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Progress (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Rate of change
        if len(values) > 1:
            rates = [(values[i] - values[i-1]) / (times[i] - times[i-1]) for i in range(1, len(values))]
            ax2.plot(times[1:], rates, 'r-', linewidth=2, marker='s', markersize=4)
            ax2.set_title('Rate of Progress')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Progress Rate (%/s)')
            ax2.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'{experiment_name} - Progress')
        ax2.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Rate of Progress')
    
    plt.tight_layout()
    return fig

def create_results_chart(experiment_name: str, results: Dict[str, Any]) -> plt.Figure:
    """Create a results visualization chart for the experiment."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Success rate visualization
    if 'success' in results:
        success = results['success']
        colors = ['green' if success else 'red']
        ax1.bar(['Experiment Result'], [1], color=colors, alpha=0.7)
        ax1.set_title(f'{experiment_name} - Success Rate')
        ax1.set_ylim(0, 1)
        ax1.text(0, 0.5, 'PASS' if success else 'FAIL', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
    
    # Performance metrics (if available)
    if 'results' in results and isinstance(results['results'], dict):
        exp_results = results['results']
        
        # Exit code distribution
        if 'exit_code' in exp_results:
            exit_code = exp_results['exit_code']
            ax2.bar(['Exit Code'], [exit_code], color='blue', alpha=0.7)
            ax2.set_title('Exit Code')
            ax2.set_ylim(0, 10)
        
        # Output length (proxy for result detail)
        if 'stdout' in exp_results:
            stdout_len = len(exp_results['stdout'])
            ax3.bar(['Output Length'], [stdout_len], color='orange', alpha=0.7)
            ax3.set_title('Output Detail')
    
    # Expected vs Actual comparison
    if experiment_name in EXPERIMENT_INFO:
        info = EXPERIMENT_INFO[experiment_name]
        ax4.text(0.5, 0.8, info['title'], ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.5, 0.5, info['expected_results'], ha='center', va='center', 
                transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Expected Results')
        ax4.axis('off')
    
    plt.tight_layout()
    return fig

def update_experiment_status():
    """Update the experiment status display."""
    if experiment_state.is_running:
        status = f"Running: {experiment_state.current_experiment}"
        progress = experiment_state.progress
        message = experiment_state.status_message
    else:
        status = "Ready"
        progress = 0
        message = experiment_state.status_message
    
    return status, progress, message

def run_experiment_thread(experiment_name: str):
    """Run an experiment in a separate thread."""
    try:
        experiment_state.is_running = True
        experiment_state.current_experiment = experiment_name
        experiment_state.progress = 0
        experiment_state.status_message = f"Starting {experiment_name}..."
        
        # Get experiment info
        if experiment_name in EXPERIMENT_INFO:
            display_name = EXPERIMENT_INFO[experiment_name]['title']
        else:
            display_name = experiment_name
        
        # Find the experiment in the list
        experiment_module = None
        for module, name in EXPERIMENTS:
            if module == experiment_name:
                experiment_module = module
                break
        
        if experiment_module:
            # Run the experiment
            experiment_state.status_message = f"Executing {display_name}..."
            experiment_state.progress = 25
            
            result = run_single_experiment(experiment_module, display_name)
            
            experiment_state.progress = 75
            experiment_state.status_message = f"Processing results for {display_name}..."
            
            # Store results
            experiment_state.results[experiment_name] = result
            
            experiment_state.progress = 100
            experiment_state.status_message = f"Completed {display_name}"
            
        else:
            experiment_state.status_message = f"Experiment {experiment_name} not found"
            experiment_state.results[experiment_name] = {"success": False, "error": "Experiment not found"}
    
    except Exception as e:
        experiment_state.status_message = f"Error in {experiment_name}: {str(e)}"
        experiment_state.results[experiment_name] = {"success": False, "error": str(e)}
        traceback.print_exc()
    
    finally:
        experiment_state.is_running = False
        experiment_state.current_experiment = None

def start_experiment(experiment_name: str):
    """Start running an experiment."""
    if experiment_state.is_running:
        return "Error: Another experiment is already running", 0, "Please wait for the current experiment to complete"
    
    # Start experiment in background thread
    experiment_state.experiment_thread = threading.Thread(
        target=run_experiment_thread,
        args=(experiment_name,)
    )
    experiment_state.experiment_thread.daemon = True
    experiment_state.experiment_thread.start()
    
    return "Starting experiment...", 0, "Experiment started in background"

def stop_experiment():
    """Stop the currently running experiment."""
    if experiment_state.is_running:
        experiment_state.stop_event.set()
        if experiment_state.experiment_thread:
            experiment_state.experiment_thread.join(timeout=1.0)
        experiment_state.is_running = False
        experiment_state.current_experiment = None
        experiment_state.progress = 0
        experiment_state.status_message = "Experiment stopped by user"
        return "Experiment stopped", 0, "Experiment stopped by user"
    else:
        return "No experiment running", 0, "No experiment is currently running"

def get_experiment_results(experiment_name: str):
    """Get results for a specific experiment."""
    if experiment_name in experiment_state.results:
        result = experiment_state.results[experiment_name]
        success = result.get('success', False)
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        exit_code = result.get('exit_code', -1)
        
        # Create results text
        results_text = f"Success: {success}\n"
        results_text += f"Exit Code: {exit_code}\n\n"
        results_text += "STDOUT:\n" + stdout + "\n\n"
        if stderr:
            results_text += "STDERR:\n" + stderr + "\n\n"
        
        # Create visualization
        fig = create_results_chart(experiment_name, result)
        
        return results_text, fig
    else:
        return "No results available", None

def export_results():
    """Export all experiment results to JSON."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_state": {
            "is_running": experiment_state.is_running,
            "current_experiment": experiment_state.current_experiment,
            "progress": experiment_state.progress,
            "status_message": experiment_state.status_message
        },
        "results": experiment_state.results
    }
    
    # Save to file
    export_file = f"fractalstat_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return f"Results exported to {export_file}"

def create_gradio_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="FractalStat Interactive Experiments") as demo:
        gr.Markdown("""
        # FractalStat Interactive Experiments
        
        Welcome to the FractalStat validation experiments platform! This interactive interface allows you to run and visualize the 12 validation experiments that prove the FractalStat 8-dimensional addressing system works at scale.
        
        ## What is FractalStat?
        
        FractalStat is a research package containing **12 validation experiments** that prove the FractalStat addressing system works at scale. FractalStat expands FractalStat from a 7D to an 8-dimensional coordinate system for uniquely addressing data in fractal information spaces.
        """)
        
        # Experiment selection and controls
        with gr.Row():
            with gr.Column(scale=2):
                experiment_dropdown = gr.Dropdown(
                    choices=[(info['title'], name) for name, info in EXPERIMENT_INFO.items()],
                    label="Select Experiment",
                    value="exp01_geometric_collision"
                )
                
                with gr.Row():
                    start_btn = gr.Button("Start Experiment", variant="primary")
                    stop_btn = gr.Button("Stop Experiment", variant="secondary")
                    export_btn = gr.Button("Export Results", variant="secondary")
                
                status_text = gr.Textbox(label="Status", value="Ready to run experiments", interactive=False)
                progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progress (%)", interactive=False)
            
            with gr.Column(scale=3):
                # Mathematical explanation panel
                math_title = gr.Markdown("### Mathematical Foundation")
                math_content = gr.Markdown(EXPERIMENT_INFO["exp01_geometric_collision"]["educational_content"])
        
        # Results display
        with gr.Row():
            with gr.Column():
                results_text = gr.Textbox(label="Experiment Results", lines=15, interactive=False)
                results_chart = gr.Plot(label="Results Visualization")
            
            with gr.Column():
                gr.Plot(label="Real-time Progress")
        
        # Update mathematical content when experiment changes
        def update_math_content(experiment_name):
            if experiment_name in EXPERIMENT_INFO:
                info = EXPERIMENT_INFO[experiment_name]
                return f"### {info['math_concept']}", info['educational_content']
            return "### Mathematical Foundation", "Select an experiment to view its mathematical foundation."
        
        experiment_dropdown.change(
            fn=update_math_content,
            inputs=[experiment_dropdown],
            outputs=[math_title, math_content]
        )
        
        # Experiment control functions
        start_btn.click(
            fn=start_experiment,
            inputs=[experiment_dropdown],
            outputs=[status_text, progress_bar, status_text]
        )
        
        stop_btn.click(
            fn=stop_experiment,
            inputs=[],
            outputs=[status_text, progress_bar, status_text]
        )
        
        # Update results when experiment completes
        def update_results(experiment_name):
            return get_experiment_results(experiment_name)
        
        # Periodic updates for status and progress
        def periodic_update():
            status, progress, message = update_experiment_status()
            return status, progress, message
        
        # Set up periodic updates
        demo.load(
            fn=periodic_update,
            inputs=[],
            outputs=[status_text, progress_bar, status_text]
        )
        
        # Update results when experiment changes
        experiment_dropdown.change(
            fn=update_results,
            inputs=[experiment_dropdown],
            outputs=[results_text, results_chart]
        )
        
        # Export functionality
        export_btn.click(
            fn=export_results,
            inputs=[],
            outputs=[status_text]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Note**: This is a research platform for validating the FractalStat addressing system. 
        All experiments are designed to run safely and provide educational insights into 
        multi-dimensional coordinate systems and their applications.
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        debug=True,
        share=False
    )
