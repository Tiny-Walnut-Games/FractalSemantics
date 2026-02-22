# Brief overview

Guidelines for developing and maintaining the FractalSemantics GUI application and related web interface components. This covers the Streamlit-based web application, progress tracking, visualization components, and user experience design.

## Project structure

- Main GUI application: `gui_app.py` (Streamlit-based web interface)
- Launch script: `launch_gui.py` (Entry point with dependency management)
- Testing: `test_gui.py` (Comprehensive GUI testing suite)
- Optimization: `optimize_gui.py` (Performance monitoring and optimization)
- Whitespace fixer: `fix_whitespace.py` (Code formatting utility)
- Requirements: `gui_requirements.txt` (GUI-specific dependencies)

## Technology stack

- Primary framework: Streamlit for web interface
- Visualization: Plotly for interactive charts and graphs
- Progress tracking: Custom progress communication system
- Styling: Custom CSS with gradient themes
- Data handling: Pandas for data manipulation
- Async operations: Asyncio for concurrent experiment execution

## Code organization

- Class-based architecture: `FractalSemanticsGUI` main application class
- Modular rendering: Separate methods for each UI component
- Session state management: Streamlit session state for data persistence
- Error handling: Graceful fallbacks and user-friendly error messages
- Progress integration: Real-time progress bars and status updates

## Progress tracking implementation

- Progress communication: Leverage existing `fractalsemantics.progress_comm` module
- Real-time updates: Streamlit auto-refresh every 5 seconds
- Progress visualization: Multiple progress components (bars, charts, tables)
- Progress callback: Enhanced callback with detailed progress data
- Fallback handling: Graceful degradation when progress messages unavailable

## User interface design

- Header: Gradient-styled main title with project description
- Sidebar: Experiment selection, configuration, and system information
- Tabs: Dashboard, Experiments, Visualizations, Education, Settings
- Progress section: Real-time progress bars, metrics, and timeline charts
- Metrics display: Success rates, performance statistics, and system health
- Educational content: Mathematical explanations and learning materials

## Visualization components

- Dashboard charts: Success rates, performance metrics, experiment timelines
- Progress timeline: Interactive Plotly line chart with real-time updates
- Performance analysis: Scatter plots, bar charts, and distribution histograms
- Educational content: Expandable sections with mathematical explanations
- Data export: JSON, CSV, and PDF export capabilities

## Experiment integration

- Batch execution: Parallel and sequential experiment running
- Progress monitoring: Real-time progress tracking during execution
- Result display: Detailed experiment results with metrics and educational content
- Error handling: Clear error messages and troubleshooting guidance
- Educational focus: Each experiment includes mathematical concepts and real-world applications

## Performance optimization

- Resource monitoring: CPU, memory, and disk space tracking
- Dependency validation: Automatic checking of required packages
- Performance benchmarking: NumPy and Pandas operation timing
- Optimization recommendations: System-specific performance suggestions
- Configuration management: TOML-based optimization settings

## Testing strategy

- Import validation: Verify all required dependencies are available
- Component testing: Test individual GUI components and rendering
- Integration testing: Test experiment runner integration
- Performance testing: Validate response times and resource usage
- File structure validation: Ensure all required files are present

## Code quality standards

- Whitespace management: No whitespace on empty lines (use fix_whitespace.py)
- Error handling: Comprehensive try-catch blocks with user-friendly messages
- Documentation: Detailed docstrings for all methods and classes
- Type hints: Full type annotation for better code maintainability
- Session state: Proper management of Streamlit session state variables

## Development workflow

- Testing first: Always run test_gui.py before committing changes
- Performance monitoring: Use optimize_gui.py to check system resources
- Progress integration: Ensure new experiments integrate with progress system
- Educational content: Include mathematical explanations for all experiments
- Cross-platform compatibility: Test on different operating systems and browsers

## Error handling patterns

- Graceful degradation: UI continues working even if experiments fail
- User feedback: Clear status messages and progress indicators
- Logging: Comprehensive logging for debugging and monitoring
- Recovery: Automatic recovery from transient errors and timeouts
- Validation: Input validation and boundary checking

## Maintenance guidelines

- Regular testing: Run test suite before each deployment
- Performance monitoring: Monitor resource usage and response times
- Dependency updates: Keep dependencies up-to-date with security patches
- Code cleanup: Regular whitespace and formatting fixes
- Documentation: Keep README and documentation current with changes
