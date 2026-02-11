#!/usr/bin/env python3
"""
FractalSemantics GUI Application

A comprehensive web-based GUI for running and visualizing FractalSemantics experiments.
Provides real-time monitoring, interactive dashboards, and educational content display.

Features:
- Real-time experiment execution and monitoring
- Interactive data visualization with Plotly
- Educational content and mathematical explanations
- Batch experiment management
- Results export and reporting
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fractalsemantics.experiment_runner import (
    BatchRunResult,
    ExperimentResult,
    ExperimentRunner,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="FractalSemantics GUI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .experiment-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .error-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FractalSemanticsGUI:
    """Main GUI application for FractalSemantics experiments."""
    
    def __init__(self):
        self.runner = ExperimentRunner()
        self.experiment_results: List[ExperimentResult] = []
        self.batch_result: Optional[BatchRunResult] = None
        self.current_experiment: Optional[ExperimentResult] = None
        self.is_running = False
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'experiment_results' not in st.session_state:
            st.session_state.experiment_results = []
        if 'batch_result' not in st.session_state:
            st.session_state.batch_result = None
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'current_experiment_id' not in st.session_state:
            st.session_state.current_experiment_id = None
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = []
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🔬 FractalSemantics Experiment Suite</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; color: #6c757d; font-size: 1.1rem;">
                Interactive GUI for running and visualizing FractalSemantics experiments
                <br><span style="font-size: 0.9rem;">8-dimensional addressing system validation suite</span>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with experiment selection and controls."""
        st.sidebar.title("🔬 Experiment Control")
        
        # Experiment selection
        st.sidebar.subheader("Experiment Selection")
        
        experiment_options = list(self.runner.experiment_configs.keys())
        selected_experiments = st.sidebar.multiselect(
            "Select Experiments",
            experiment_options,
            default=["EXP-01", "EXP-02", "EXP-03"],
            help="Choose which experiments to run"
        )
        
        # Configuration options
        st.sidebar.subheader("Configuration")
        
        quick_mode = st.sidebar.checkbox(
            "Quick Mode",
            value=True,
            help="Run experiments with smaller sample sizes for faster results"
        )
        
        parallel_mode = st.sidebar.checkbox(
            "Parallel Execution",
            value=True,
            help="Run experiments in parallel for faster completion"
        )
        
        # Feature level selection
        feature_level = st.sidebar.selectbox(
            "Feature Level",
            ["Quick", "Full"],
            index=0,
            help="Quick mode for development, Full mode for comprehensive validation"
        )
        
        # Action buttons
        st.sidebar.subheader("Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            run_button = st.button(
                "🚀 Run Experiments",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_running
            )
        
        with col2:
            stop_button = st.button(
                "⏹️ Stop",
                type="secondary",
                use_container_width=True,
                disabled=not st.session_state.is_running
            )
        
        # Export options
        st.sidebar.subheader("Export Options")
        
        export_format = st.sidebar.selectbox(
            "Export Format",
            ["JSON", "CSV", "PDF"],
            index=0
        )
        
        if st.sidebar.button("📊 Export Results", use_container_width=True):
            self.export_results(export_format)
        
        # System info
        st.sidebar.subheader("System Information")
        st.sidebar.info(f"""
        **Status**: {"Running" if st.session_state.is_running else "Idle"}
        **Experiments**: {len(st.session_state.experiment_results)}
        **Python**: {sys.version.split()[0]}
        """)
        
        return selected_experiments, quick_mode, parallel_mode, feature_level, run_button, stop_button
    
    def render_main_content(self):
        """Render the main content area."""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "🔬 Experiments", 
            "📈 Visualizations", 
            "📚 Education", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_experiment_details()
        
        with tab3:
            self.render_visualizations()
        
        with tab4:
            self.render_education()
        
        with tab5:
            self.render_settings()
    
    def render_dashboard(self):
        """Render the main dashboard with overall statistics."""
        st.subheader("📊 Experiment Dashboard")
        
        if not st.session_state.experiment_results:
            st.info("No experiments have been run yet. Select experiments in the sidebar and click 'Run Experiments' to get started.")
            return
        
        # Overall statistics
        results = st.session_state.experiment_results
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r.success)
        failed_experiments = total_experiments - successful_experiments
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Experiments",
                total_experiments,
                delta=f"+{total_experiments}"
            )
        
        with col2:
            st.metric(
                "Successful",
                successful_experiments,
                delta=f"+{successful_experiments}",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Failed",
                failed_experiments,
                delta=f"+{failed_experiments}",
                delta_color="inverse"
            )
        
        with col4:
            success_rate = (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                delta=f"+{success_rate:.1f}%"
            )
        
        # Progress visualization
        st.subheader("Experiment Progress")
        self.render_progress_chart()
        
        # Performance metrics
        st.subheader("Performance Metrics")
        self.render_performance_metrics()
    
    def render_experiment_details(self):
        """Render detailed experiment information."""
        st.subheader("🔬 Experiment Details")
        
        if not st.session_state.experiment_results:
            st.info("No experiments to display. Run some experiments first!")
            return
        
        # Filter and sort experiments
        experiment_filter = st.selectbox(
            "Filter by Experiment",
            ["All"] + list(self.runner.experiment_configs.keys())
        )
        
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Success", "Failed", "Warning", "Partial Success"]
        )
        
        filtered_results = st.session_state.experiment_results
        
        if experiment_filter != "All":
            filtered_results = [r for r in filtered_results if r.experiment_id == experiment_filter]
        
        if status_filter != "All":
            if status_filter == "Success":
                filtered_results = [r for r in filtered_results if r.result_type == "success"]
            elif status_filter == "Failed":
                filtered_results = [r for r in filtered_results if r.result_type == "failure"]
            elif status_filter == "Warning":
                filtered_results = [r for r in filtered_results if r.result_type == "warning"]
            elif status_filter == "Partial Success":
                filtered_results = [r for r in filtered_results if r.result_type == "partial_success"]
        
        # Display experiments
        for result in filtered_results:
            with st.container():
                st.markdown(f'<div class="experiment-card">', unsafe_allow_html=True)
                
                # Header
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### {result.experiment_id}")
                    st.markdown(f"**{self.runner.experiment_configs[result.experiment_id]['description']}**")
                
                with col2:
                    if result.success:
                        st.markdown('<span class="success-badge">✅ Success</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="error-badge">❌ Failed</span>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**Duration:** {result.duration:.2f}s")
                
                # Details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    for key, value in result.metrics.items():
                        st.write(f"- {key}: {value}")
                
                with col2:
                    st.markdown("**🎯 Result Type:**")
                    st.write(f"- {result.result_type.title()}")
                    st.write(f"- Educational Content: {len(result.educational_content)} sections")
                
                # Output and educational content
                with st.expander("View Experiment Output"):
                    st.code(result.output, language="text")
                
                with st.expander("View Educational Content"):
                    for i, content in enumerate(result.educational_content, 1):
                        st.markdown(f"**Section {i}:**")
                        st.markdown(content)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")
    
    def render_visualizations(self):
        """Render interactive data visualizations."""
        st.subheader("📈 Data Visualizations")
        
        if not st.session_state.experiment_results:
            st.info("No data to visualize. Run some experiments first!")
            return
        
        results = st.session_state.experiment_results
        
        # Experiment success rate by type
        st.markdown("### Experiment Success Rate by Type")
        success_data = []
        for exp_id in self.runner.experiment_configs.keys():
            exp_results = [r for r in results if r.experiment_id == exp_id]
            if exp_results:
                success_rate = sum(1 for r in exp_results if r.success) / len(exp_results)
                success_data.append({
                    'Experiment': exp_id,
                    'Success Rate': success_rate * 100,
                    'Count': len(exp_results)
                })
        
        if success_data:
            fig = px.bar(
                success_data,
                x='Experiment',
                y='Success Rate',
                color='Success Rate',
                color_continuous_scale='RdYlGn',
                title="Success Rate by Experiment Type",
                text='Count'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance analysis
        st.markdown("### Performance Analysis")
        perf_data = []
        for result in results:
            perf_data.append({
                'Experiment': result.experiment_id,
                'Duration': result.duration,
                'Success': 'Success' if result.success else 'Failed',
                'Result Type': result.result_type
            })
        
        if perf_data:
            fig = px.scatter(
                perf_data,
                x='Duration',
                y='Experiment',
                color='Success',
                size='Duration',
                title="Experiment Duration Analysis",
                hover_data=['Result Type']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Educational content analysis
        st.markdown("### Educational Content Analysis")
        edu_data = []
        for result in results:
            edu_data.append({
                'Experiment': result.experiment_id,
                'Content Sections': len(result.educational_content),
                'Success': 'Success' if result.success else 'Failed'
            })
        
        if edu_data:
            fig = px.bar(
                edu_data,
                x='Experiment',
                y='Content Sections',
                color='Success',
                title="Educational Content by Experiment",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_education(self):
        """Render educational content and explanations."""
        st.subheader("📚 FractalSemantics Education")
        
        st.markdown("""
        ## What is FractalSemantics?
        
        FractalSemantics is a research package containing **12 validation experiments** that prove the FractalSemantics 8-dimensional addressing system works at scale.
        
        ### The 8 Dimensions:
        
        1. **Realm** - Domain classification (data, narrative, system, etc.)
        2. **Lineage** - Generation from LUCA (Last Universal Common Ancestor)
        3. **Temperature** - Thermal activity level (0.0 to abs(velocity) * density)
        4. **Adjacency** - Relational neighbors (graph connections)
        5. **Horizon** - Lifecycle stage (genesis, emergence, peak, decay, crystallization)
        6. **Resonance** - Charge/alignment (-1.0 to 1.0)
        7. **Velocity** - Rate of change
        8. **Density** - Compression distance (0.0 to 1.0)
        9. **Alignment** - Value based on alignment map
        
        ### Mathematical Foundations
        
        FractalSemantics uses advanced mathematical concepts including:
        
        - **8-Dimensional Coordinate Space**
        - **Collision Resistance Mathematics**
        - **Fractal Geometry Principles**
        - **Information Theory**
        - **Statistical Mechanics**
        - **Topological Conservation Laws**
        """)
        
        # Show educational content from recent experiments
        if st.session_state.experiment_results:
            st.markdown("## Recent Educational Content")
            
            # Get the most recent experiment
            latest_result = st.session_state.experiment_results[-1]
            
            st.markdown(f"### {latest_result.experiment_id} - {self.runner.experiment_configs[latest_result.experiment_id]['educational_focus']}")
            
            for i, content in enumerate(latest_result.educational_content, 1):
                with st.expander(f"Section {i}"):
                    st.markdown(content)
    
    def render_settings(self):
        """Render application settings."""
        st.subheader("⚙️ Application Settings")
        
        # Auto-refresh settings
        st.markdown("### Auto-Refresh Settings")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=30,
                value=5
            )
            st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")
        
        # Display settings
        st.markdown("### Display Settings")
        show_detailed_output = st.checkbox("Show Detailed Experiment Output", value=True)
        show_educational_content = st.checkbox("Show Educational Content", value=True)
        
        # Reset data
        st.markdown("### Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Results", type="secondary"):
                st.session_state.experiment_results = []
                st.session_state.batch_result = None
                st.rerun()
        
        with col2:
            if st.button("💾 Save Session", type="primary"):
                self.save_session()
    
    def render_progress_chart(self):
        """Render real-time progress visualization."""
        if st.session_state.is_running and st.session_state.progress_data:
            progress_data = st.session_state.progress_data
            
            fig = go.Figure()
            
            # Add progress line
            fig.add_trace(go.Scatter(
                x=[p['time'] for p in progress_data],
                y=[p['progress'] for p in progress_data],
                mode='lines+markers',
                name='Progress',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Real-time Experiment Progress",
                xaxis_title="Time",
                yaxis_title="Progress (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No progress data available. Start an experiment to see real-time progress.")
    
    def render_performance_metrics(self):
        """Render performance metrics visualization."""
        if not st.session_state.experiment_results:
            st.info("No performance data available.")
            return
        
        results = st.session_state.experiment_results
        
        # Calculate metrics
        durations = [r.duration for r in results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Duration Distribution", "Success Rate by Experiment", "Performance Timeline", "Result Type Distribution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Duration distribution
        fig.add_trace(
            go.Histogram(x=durations, name="Duration", nbinsx=10),
            row=1, col=1
        )
        
        # Success rate by experiment
        success_rates = []
        exp_names = []
        for exp_id in self.runner.experiment_configs.keys():
            exp_results = [r for r in results if r.experiment_id == exp_id]
            if exp_results:
                rate = sum(1 for r in exp_results if r.success) / len(exp_results)
                success_rates.append(rate * 100)
                exp_names.append(exp_id)
        
        fig.add_trace(
            go.Bar(x=exp_names, y=success_rates, name="Success Rate"),
            row=1, col=2
        )
        
        # Performance timeline
        timestamps = list(range(len(results)))
        fig.add_trace(
            go.Scatter(x=timestamps, y=durations, mode='lines+markers', name="Duration"),
            row=2, col=1
        )
        
        # Result type distribution
        result_types = [r.result_type for r in results]
        type_counts = {}
        for rt in result_types:
            type_counts[rt] = type_counts.get(rt, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(type_counts.keys()), values=list(type_counts.values()), name="Result Types"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Duration", f"{avg_duration:.2f}s")
        
        with col2:
            st.metric("Min Duration", f"{min_duration:.2f}s")
        
        with col3:
            st.metric("Max Duration", f"{max_duration:.2f}s")
        
        with col4:
            st.metric("Total Runtime", f"{sum(durations):.2f}s")
    
    def export_results(self, format_type: str):
        """Export experiment results to various formats."""
        if not st.session_state.experiment_results:
            st.error("No results to export. Run some experiments first!")
            return
        
        results = st.session_state.experiment_results
        
        if format_type == "JSON":
            # Export as JSON
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_experiments": len(results),
                "successful_experiments": sum(1 for r in results if r.success),
                "failed_experiments": sum(1 for r in results if not r.success),
                "experiment_results": [asdict(r) for r in results]
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"fractalsemantics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        elif format_type == "CSV":
            # Export as CSV
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for result in results:
                row = {
                    'experiment_id': result.experiment_id,
                    'success': result.success,
                    'duration': result.duration,
                    'result_type': result.result_type,
                    'educational_content_count': len(result.educational_content),
                    'output_length': len(result.output)
                }
                # Add metrics as separate columns
                for key, value in result.metrics.items():
                    row[f'metric_{key}'] = value
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name=f"fractalsemantics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif format_type == "PDF":
            # Export as PDF (simplified - would need additional libraries for full PDF generation)
            st.info("PDF export would require additional libraries. Consider using the JSON or CSV export for now.")
    
    def save_session(self):
        """Save current session state to a file."""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_results": [asdict(r) for r in st.session_state.experiment_results],
            "batch_result": asdict(st.session_state.batch_result) if st.session_state.batch_result else None
        }
        
        # Save to results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        st.success(f"Session saved to {filepath}")
    
    async def run_experiments_async(self, experiment_ids: List[str], quick_mode: bool, parallel_mode: bool):
        """Run experiments asynchronously with real-time updates."""
        st.session_state.is_running = True
        st.session_state.progress_data = []
        
        try:
            # Run experiments
            batch_result = await self.runner.run_batch_experiments(
                experiment_ids=experiment_ids,
                quick_mode=quick_mode,
                parallel=parallel_mode,
                progress_callback=self.progress_callback
            )
            
            # Update session state
            st.session_state.experiment_results.extend(batch_result.experiment_results)
            st.session_state.batch_result = batch_result
            
            st.success(f"Batch run completed! {batch_result.successful_experiments}/{batch_result.total_experiments} experiments successful.")
            
        except Exception as e:
            st.error(f"Error running experiments: {str(e)}")
        
        finally:
            st.session_state.is_running = False
    
    def progress_callback(self, current: int, total: int, result: ExperimentResult):
        """Callback function for experiment progress updates."""
        progress_percent = (current / total) * 100
        
        # Add to progress data
        st.session_state.progress_data.append({
            'time': datetime.now().isoformat(),
            'progress': progress_percent,
            'current': current,
            'total': total,
            'experiment': result.experiment_id
        })
        
        # Keep only last 100 data points
        if len(st.session_state.progress_data) > 100:
            st.session_state.progress_data = st.session_state.progress_data[-100:]
    
    def run(self):
        """Main application loop."""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        selected_experiments, quick_mode, parallel_mode, feature_level, run_button, stop_button = self.render_sidebar()
        
        # Handle experiment execution
        if run_button and selected_experiments:
            if st.session_state.is_running:
                st.warning("Experiments are already running!")
            else:
                st.info(f"Starting {len(selected_experiments)} experiments in {'parallel' if parallel_mode else 'sequential'} mode...")
                
                # Run experiments in a separate thread to avoid blocking
                asyncio.run(self.run_experiments_async(
                    selected_experiments,
                    quick_mode,
                    parallel_mode
                ))
        
        if stop_button:
            st.session_state.is_running = False
            st.info("Experiment execution stopped.")
        
        # Render main content
        self.render_main_content()

def main():
    """Main entry point for the GUI application."""
    try:
        gui = FractalSemanticsGUI()
        gui.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"GUI application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()