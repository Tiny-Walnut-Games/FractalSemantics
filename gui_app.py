#!/usr/bin/env python3
"""
FractalSemantics GUI Application 🔬

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
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

import gui_state_manager
from fractalsemantics.progress_comm import clear_progress_file, read_progress_from_file

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fractalsemantics.experiment_runner import (
    BatchRunResult,
    ExperimentResult,
    ExperimentRunner,
)
from gui_state_manager import (
    cleanup_session_state,
    ensure_session_state_initialized,
    get_session_state_snapshot,
    safe_session_state_update,
    validate_session_state,
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


# Create results directory if it doesn't exist
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Set progress file path - experiments write to project root/results/ directory
progress_file = results_dir / "gui_progress.jsonl"
os.environ["FRACTALSEMANTICS_PROGRESS_FILE"] = str(progress_file)

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

# Initialize session state at the very beginning to prevent access errors
try:
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
except Exception as e:
    # Fallback initialization if session state is corrupted
    st.session_state.experiment_results = []
    st.session_state.batch_result = None
    st.session_state.is_running = False
    st.session_state.current_experiment_id = None
    st.session_state.progress_data = []
    logger.error(f"Global session state initialization error: {e}")

class FractalSemanticsGUI:
    """Main GUI application for FractalSemantics experiments."""

    def __init__(self):
        self.runner = ExperimentRunner()
        self.experiment_results: list[ExperimentResult] = []
        self.batch_result: Optional[BatchRunResult] = None
        self.current_experiment: Optional[ExperimentResult] = None
        self.is_running = False
        self.setup_session_state()

    def setup_session_state(self):
        """Initialize session state variables."""
        try:
            # Initialize session state with thread-safe operations
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

            # Set up progress file environment variable for subprocess communication
            import os
            from pathlib import Path

            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            # Set progress file path
            progress_file = results_dir / "gui_progress.jsonl"
            os.environ["FRACTALSEMANTICS_PROGRESS_FILE"] = str(progress_file)

        except Exception as e:
            # Fallback initialization if session state is corrupted
            st.session_state.experiment_results = []
            st.session_state.batch_result = None
            st.session_state.is_running = False
            st.session_state.current_experiment_id = None
            st.session_state.progress_data = []
            logger.error(f"Session state initialization error: {e}")

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

        # Select all button
        col1, col2 = st.sidebar.columns(2)
        with col1:
            select_all = st.button("Select All", key="select_all_experiments")
        with col2:
            clear_all = st.button("Clear All", key="clear_all_experiments")

        # Initialize default selection
        if 'selected_experiments' not in st.session_state:
            st.session_state.selected_experiments = ["EXP-01", "EXP-02", "EXP-03"]

        # Handle select all/clear all
        if select_all:
            st.session_state.selected_experiments = experiment_options
        elif clear_all:
            st.session_state.selected_experiments = []

        selected_experiments = st.sidebar.multiselect(
            "Select Experiments",
            experiment_options,
            default=st.session_state.selected_experiments,
            help="Choose which experiments to run"
        )

        # Update session state with current selection
        st.session_state.selected_experiments = selected_experiments

        # Configuration options
        st.sidebar.subheader("Configuration")

        # Combined feature level selection
        feature_level = st.sidebar.selectbox(
            "Feature Level",
            ["Quick", "Full"],
            index=0,
            help="Quick mode for development, Full mode for comprehensive validation"
        )

        parallel_mode = st.sidebar.checkbox(
            "Parallel Execution",
            value=True,
            help="Run experiments in parallel for faster completion"
        )

        # Quick mode is determined by feature level
        quick_mode = feature_level == "Quick"

        # Action buttons
        st.sidebar.subheader("Actions")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            run_button = st.button(
                "▶️ Run Experiments",
                type="primary",
                width="stretch",
                disabled=st.session_state.is_running
            )

        with col2:
            stop_button = st.button(
                "⏹️ Stop",
                type="secondary",
                width="stretch",
                disabled=not st.session_state.is_running
            )

        # Export options
        st.sidebar.subheader("Export Options")

        export_format = st.sidebar.selectbox(
            "Export Format",
            ["JSON", "CSV", "PDF"],
            index=0
        )

        if st.sidebar.button("chart Export Results", width="stretch"):
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
        st.subheader("🧪 Experiment Dashboard")

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
                st.markdown('<div class="experiment-card">', unsafe_allow_html=True)

                # Header
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"### {result.experiment_id}")
                    exp_config = self.runner.experiment_configs[result.experiment_id]
                    st.markdown(f"**{exp_config.description}**")

                with col2:
                    # Display appropriate badge based on result type
                    result_type = result.result_type or "unknown"
                    if result_type == "success":
                        st.markdown('<span class="success-badge">✅ Success</span>', unsafe_allow_html=True)
                    elif result_type == "warning":
                        st.markdown('<span class="warning-badge">⚠️ Warning</span>', unsafe_allow_html=True)
                    elif result_type == "partial_success":
                        st.markdown('<span class="warning-badge">⚠️ Partial Success</span>', unsafe_allow_html=True)
                    elif result_type == "failure":
                        st.markdown('<span class="error-badge">❌ Failed</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="warning-badge">❓ {result_type.title()}</span>', unsafe_allow_html=True)

                with col3:
                    st.markdown(f"**Duration:** {result.duration:.2f}s")

                # Details
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    for key, value in result.metrics.items():
                        st.write(f"- {key}: {value}")

                with col2:
                    st.markdown("**❓ Result Type:**")
                    result_type = result.result_type or "unknown"
                    st.write(f"- {result_type.title()}")
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
        st.markdown("### 🧪 Experiment Success Rate by Type")
        success_data = []
        for exp_id in self.runner.experiment_configs:
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
            st.plotly_chart(fig, width="stretch")

        # Performance analysis
        st.markdown("### ⏱️ Performance Analysis")
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
            st.plotly_chart(fig, width="stretch")

        # Educational content analysis
        st.markdown("### 📚 Educational Content Analysis")
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
            st.plotly_chart(fig, width="stretch")

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

            exp_config = self.runner.experiment_configs[latest_result.experiment_id]
            st.markdown(f"### {latest_result.experiment_id} - {exp_config.educational_focus}")

            for i, content in enumerate(latest_result.educational_content, 1):
                with st.expander(f"Section {i}"):
                    st.markdown(content)

    def render_settings(self):
        """Render application settings."""
        st.subheader("⚙️ Application Settings")

        # Auto-refresh settings
        st.markdown("### 🔄 Auto-Refresh Settings")
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
        st.markdown("### 🖥️ Display Settings")
        st.checkbox("Show Detailed Experiment Output", value=True)
        st.checkbox("Show Educational Content", value=True)

        # Reset data
        st.markdown("### 🗄️ Data Management")
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
        """Render real-time progress visualization with individual progress bars per experiment."""
        try:
            # Always show progress chart, whether running or not
            st.markdown("### 📊 Experiment Progress")

            # Get progress file path
            progress_file = Path(os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE", "results/gui_progress.jsonl"))

            # Read current progress from file
            experiment_progress = {}
            if progress_file.exists():
                try:
                    with open(progress_file, encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    message = json.loads(line.strip())
                                    if all(field in message for field in ['experiment_id', 'progress', 'stage']):
                                        exp_id = message['experiment_id']
                                        progress_value = message['progress']

                                        # Only update progress for actual progress messages (progress >= 0)
                                        if progress_value >= 0:
                                            # Validate and clamp progress value to prevent negative values
                                            progress_value = max(0.0, min(100.0, float(progress_value)))

                                            experiment_progress[exp_id] = {
                                                'progress': progress_value,
                                                'stage': message.get('stage', 'Running'),
                                                'message': message.get('message', ''),
                                                'timestamp': message.get('timestamp', '')
                                            }
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.error(f"Error reading progress file: {e}")

            if experiment_progress:
                # Display individual progress bars for each experiment
                for exp_id, progress_info in sorted(experiment_progress.items()):
                    try:
                        progress_value = progress_info['progress']
                        stage = progress_info['stage']
                        message = progress_info['message']

                        # Create container for each experiment's progress
                        with st.container():
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.markdown(f"**{exp_id}**")
                                st.progress(progress_value / 100.0)

                            with col2:
                                st.metric("", f"{progress_value:.1f}%", label_visibility="collapsed")

                            # Show stage/message if available
                            if message:
                                st.caption(f"{stage}: {message}")

                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Error rendering progress for {exp_id}: {e}")
                        continue

                # Detailed progress chart
                st.markdown("### 📈 Progress Timeline")
                fig = go.Figure()

                # Read all progress data for timeline
                progress_data = []
                if progress_file.exists():
                    try:
                        with open(progress_file, encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        message = json.loads(line.strip())
                                        if 'timestamp' in message and 'progress' in message:
                                            progress_data.append({
                                                'time': message['timestamp'],
                                                'progress': message['progress'],
                                                'experiment': message.get('experiment_id', 'Unknown')
                                            })
                                    except json.JSONDecodeError:
                                        continue
                    except Exception:
                        pass

                if progress_data:
                    # Add progress line
                    fig.add_trace(go.Scatter(
                        x=[p['time'] for p in progress_data],
                        y=[p['progress'] for p in progress_data],
                        mode='lines+markers',
                        name='Progress',
                        line={'color': '#667eea', 'width': 3},
                        marker={'size': 8}
                    ))

                    fig.update_layout(
                        title="Real-time Experiment Progress",
                        xaxis_title="Time",
                        yaxis_title="Progress (%)",
                        height=300,
                        yaxis={'range': [0, 100]}
                    )

                    st.plotly_chart(fig, width="stretch")

                # Progress details table
                st.markdown("### 📋 Progress Details")
                progress_df = []
                for p in progress_data[-10:]:  # Show last 10 progress updates
                    try:
                        progress_df.append({
                            'Time': p.get('time', 'Unknown'),
                            'Progress': f"{p.get('progress', 0):.1f}%",
                            'Experiment': p.get('experiment', 'Unknown')
                        })
                    except Exception:
                        continue

                if progress_df:
                    import pandas as pd
                    df = pd.DataFrame(progress_df)
                    st.dataframe(df, width="stretch")
            else:
                st.info("No progress data available. Start an experiment to see real-time progress.")
        except Exception as e:
            st.error(f"Error rendering progress chart: {e}")
            logger.error(f"Progress chart rendering error: {e}")

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
        for exp_id in self.runner.experiment_configs:
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
        st.plotly_chart(fig, width="stretch")

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

    def run_experiments_sync(self, experiment_ids: list[str], quick_mode: bool, parallel_mode: bool):
        """Run experiments with real-time progress updates using background thread + polling."""
        # Ensure session state is properly initialized before any access
        ensure_session_state_initialized()

        st.session_state.is_running = True
        st.session_state.progress_data = []

        # Get progress file path
        progress_file = Path(os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE", "results/gui_progress.jsonl"))

        # Clear progress file at start
        clear_progress_file(progress_file)

        # Create dedicated progress display area
        progress_container = st.container()

        with progress_container:
            st.markdown("### - Running Experiments")
            st.markdown(f"**Selected Experiments:** {', '.join(experiment_ids)}")
            st.markdown(f"**Mode:** {'Quick' if quick_mode else 'Full'} | **Execution:** {'Parallel' if parallel_mode else 'Sequential'}")
            st.markdown("---")

            # Create placeholders for individual experiment progress bars
            st.markdown("#### Experiment Progress")
            experiment_progress_bars = {}
            for exp_id in experiment_ids:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{exp_id}**")
                        experiment_progress_bars[exp_id] = {
                            'progress_bar': col1.empty(),
                            'metric': col2.empty()
                        }

            st.markdown("---")

            # Create placeholders for overall status
            overall_progress_bar = st.empty()
            progress_metrics = st.empty()
            experiment_status = st.empty()

        # Thread-safe structures for background execution
        import threading
        import time

        experiment_complete = threading.Event()
        progress_lock = threading.Lock()
        batch_result_holder = [None]  # List to hold result (mutable)
        exception_holder = [None]  # List to hold exceptions

        # Track latest progress per experiment using thread-safe data structures
        experiment_progress = {}
        completed_experiments = set()

        def run_experiments_background():
            """Run experiments in background thread with proper asyncio handling."""
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Define progress callback that uses thread-safe data structures
                    def progress_callback(current: int, total: int, result: ExperimentResult):
                        with progress_lock:
                            completed_experiments.add(result.experiment_id)
                            experiment_progress[result.experiment_id] = {
                                'progress': 100.0,
                                'completed': True,
                                'success': result.success
                            }

                    # Run batch experiments using the thread's event loop
                    batch_result = loop.run_until_complete(
                        self.runner.run_batch_experiments(
                            experiment_ids=experiment_ids,
                            quick_mode=quick_mode,
                            parallel=parallel_mode,
                            progress_callback=progress_callback
                        )
                    )

                    with progress_lock:
                        batch_result_holder[0] = batch_result

                finally:
                    # Clean up the event loop
                    loop.close()
                    asyncio.set_event_loop(None)

            except Exception as e:
                with progress_lock:
                    exception_holder[0] = e

            finally:
                experiment_complete.set()

        # Start experiments in background thread
        experiment_thread = threading.Thread(target=run_experiments_background, daemon=True)
        experiment_thread.start()

        # Poll for progress updates in main thread
        last_progress_check = {}
        update_counter = 0
        max_poll_time = 600  # 10 minute timeout to allow for longer experiments
        start_poll_time = time.time()

        try:
            while not experiment_complete.is_set():
                # Check for timeout
                if time.time() - start_poll_time > max_poll_time:
                    st.error("❌ Experiment execution timed out after 10 minutes")
                    break

                # Read latest progress from progress file (JSONL format)
                try:
                    # Read all progress messages from the file
                    progress_messages = []
                    if progress_file.exists():
                        try:
                            with open(progress_file, encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        try:
                                            message = json.loads(line.strip())
                                            progress_messages.append(message)
                                        except json.JSONDecodeError:
                                            continue
                        except Exception:
                            pass

                    # Process progress messages in reverse order (most recent first)
                    # to get the latest progress for each experiment
                    experiment_progress_updates = {}
                    for message in reversed(progress_messages):
                        try:
                            if all(field in message for field in ['timestamp', 'experiment_id', 'progress_percent', 'stage']):
                                exp_id = message['experiment_id']
                                progress_value = message['progress_percent']
                                message_type = message.get('message_type', 'progress')

                                # Only update progress for actual progress messages, not status messages
                                # Status messages have progress = -1.0 and should not affect progress bars
                                if message_type == 'progress' and progress_value >= 0:
                                    # Validate progress value to prevent negative values
                                    progress_value = max(0.0, min(100.0, float(progress_value)))

                                    # Update progress tracking using thread-safe operations
                                    with progress_lock:
                                        if exp_id not in completed_experiments:
                                            experiment_progress[exp_id] = {
                                                'progress': progress_value,
                                                'completed': False,
                                                'stage': message.get('stage', 'Running'),
                                                'message': message.get('message', ''),
                                                'message_type': message_type
                                            }
                                elif message_type == 'complete':
                                    # Handle completion messages
                                    with progress_lock:
                                        completed_experiments.add(exp_id)
                                        experiment_progress[exp_id] = {
                                            'progress': 100.0,
                                            'completed': True,
                                            'stage': message.get('stage', 'Complete'),
                                            'message': message.get('message', ''),
                                            'message_type': message_type,
                                            'success': True  # Assume success unless marked otherwise
                                        }
                                elif message_type in ['status', 'warning', 'error']:
                                    # For status messages (progress = -1.0), only update stage/message if experiment is still running
                                    with progress_lock:
                                        if exp_id not in completed_experiments and exp_id in experiment_progress:
                                            # Update existing progress with new stage/message but keep current progress value
                                            current_progress = experiment_progress[exp_id].get('progress', 0)
                                            experiment_progress[exp_id].update({
                                                'stage': message.get('stage', 'Running'),
                                                'message': message.get('message', ''),
                                                'message_type': message_type
                                            })
                        except Exception:
                            continue

                except Exception as e:
                    # Log error but don't crash the polling
                    logger.error(f"Error reading progress: {e}")
                    pass

                # Update UI with current progress (only in main thread)
                with progress_lock:
                    for exp_id, progress_info in experiment_progress.items():
                        if exp_id in experiment_progress_bars:
                            progress_value = progress_info['progress']

                            # Only update progress bar for actual progress updates (progress >= 0)
                            # Skip non-progress messages (progress = -1.0)
                            if progress_value >= 0:
                                # Update progress bar with error handling for WebSocket disconnections
                                try:
                                    with experiment_progress_bars[exp_id]['progress_bar']:
                                        st.progress(progress_value / 100.0)
                                except Exception:
                                    # WebSocket disconnected - skip UI update
                                    pass

                                # Update metric with error handling
                                try:
                                    with experiment_progress_bars[exp_id]['metric']:
                                        if progress_info.get('completed'):
                                            status_icon = "success" if progress_info.get('success') else "error"
                                            st.metric(exp_id, f"{status_icon} Done")
                                        else:
                                            st.metric(exp_id, f"{progress_value:.1f}%")
                                except Exception:
                                    # WebSocket disconnected - skip UI update
                                    pass
                            else:
                                # For non-progress messages (progress = -1.0), only update metric if completed
                                try:
                                    with experiment_progress_bars[exp_id]['metric']:
                                        if progress_info.get('completed'):
                                            status_icon = "success" if progress_info.get('success') else "error"
                                            st.metric(exp_id, f"{status_icon} Done")
                                except Exception:
                                    # WebSocket disconnected - skip UI update
                                    pass

                    # Update overall progress with error handling
                    try:
                        total_experiments = len(experiment_ids)
                        completed_count = len(completed_experiments)
                        overall_percent = (completed_count / total_experiments * 100) if total_experiments > 0 else 0

                        with overall_progress_bar:
                            st.progress(overall_percent / 100.0)

                        with progress_metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Completed", f"{completed_count}/{total_experiments}")
                            with col2:
                                # Show currently running experiments
                                running = [exp_id for exp_id, info in experiment_progress.items()
                                         if not info.get('completed')]
                                st.metric("Running", f"{len(running)}")
                            with col3:
                                st.metric("Progress", f"{overall_percent:.1f}%")
                    except Exception:
                        # WebSocket disconnected - skip UI update
                        pass

                # Sleep briefly to allow UI updates
                time.sleep(0.5)
                update_counter += 1

            # Wait for thread to complete with timeout
            experiment_thread.join(timeout=10.0)

            # Check if thread is still alive (timeout occurred)
            if experiment_thread.is_alive():
                st.error("❌ Experiment thread did not complete within timeout")
                return

            # Check for exceptions
            with progress_lock:
                if exception_holder[0]:
                    raise exception_holder[0]

                batch_result = batch_result_holder[0]

            # Update session state with results (before any UI updates)
            st.session_state.experiment_results.extend(batch_result.experiment_results)
            st.session_state.batch_result = batch_result

            # Mark as no longer running BEFORE the rerun
            st.session_state.is_running = False

            # Clear progress file
            clear_progress_file(progress_file)

            # Use st.rerun() instead of st.experimental_rerun() for better compatibility
            # Add a small delay to ensure state is properly updated
            import time
            time.sleep(0.1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error running experiments: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

        finally:
            # Ensure proper cleanup
            try:
                # Clear progress file
                clear_progress_file(progress_file)

                # Ensure thread is properly cleaned up
                if 'experiment_thread' in locals() and experiment_thread.is_alive():
                    # Thread should already be joined, but ensure cleanup
                    pass

                # Reset running state
                st.session_state.is_running = False

            except Exception as cleanup_error:
                logger.error(f"❌ Error during cleanup: {cleanup_error}")

    def progress_callback(self, current: int, total: int, result: ExperimentResult):
        """Callback function for experiment progress updates."""
        progress_percent = (current / total) * 100

        # Extract and process progress messages from experiment result
        if 'progress_messages' in result.metrics:
            progress_messages = result.metrics['progress_messages']
            for msg in progress_messages:
                # Convert experiment progress message format to GUI format
                st.session_state.progress_data.append({
                    'time': msg.get('timestamp', datetime.now().isoformat()),
                    'progress': msg.get('progress_percent', progress_percent),
                    'current': current,
                    'total': total,
                    'experiment': msg.get('experiment_id', result.experiment_id),
                    'stage': msg.get('stage', f"Experiment {result.experiment_id}"),
                    'message': msg.get('message', f"Completed {current}/{total} experiments")
                })

        # Add overall batch progress
        st.session_state.progress_data.append({
            'time': datetime.now().isoformat(),
            'progress': progress_percent,
            'current': current,
            'total': total,
            'experiment': result.experiment_id,
            'stage': "Batch Progress",
            'message': f"Completed {current}/{total} experiments"
        })

        # Keep only last 100 data points
        if len(st.session_state.progress_data) > 100:
            st.session_state.progress_data = st.session_state.progress_data[-100:]

    def parse_experiment_progress(self, stderr_output: str):
        """Parse progress messages from experiment stderr output."""

        # Look for progress messages in the stderr output
        lines = stderr_output.split('\n')

        for line in lines:
            if line.startswith("__PROGRESS__:"):
                try:
                    # Extract JSON part after the marker
                    json_str = line[len("__PROGRESS__:"):]
                    data = json.loads(json_str)

                    # Validate required fields
                    if all(field in data for field in ['timestamp', 'experiment_id', 'progress_percent', 'stage']):
                        # Add to progress data
                        st.session_state.progress_data.append({
                            'time': data['timestamp'],
                            'progress': data['progress_percent'],
                            'experiment': data['experiment_id'],
                            'stage': data['stage'],
                            'message': data.get('message', ''),
                            'current': data.get('current', 0),
                            'total': data.get('total', 100)
                        })

                        # Keep only last 100 data points
                        if len(st.session_state.progress_data) > 100:
                            st.session_state.progress_data = st.session_state.progress_data[-100:]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

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
                # Run experiments synchronously with proper error handling
                self.run_experiments_sync(
                    selected_experiments,
                    quick_mode,
                    parallel_mode
                )
                # Experiments complete - results will display naturally

        if stop_button:
            st.session_state.is_running = False
            st.info("Experiment execution stopped.")

        # Render main content
        self.render_main_content()

def main():
    """Main entry point for the GUI application."""
    try:
        # Ensure session state is initialized BEFORE any Streamlit operations
        gui_state_manager.ensure_session_state_initialized()

        # Create GUI instance (this will call setup_session_state again, but it's safe)
        gui = FractalSemanticsGUI()
        gui.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"GUI application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
