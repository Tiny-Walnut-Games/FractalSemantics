#!/usr/bin/env python3
"""
FractalSemantics GUI Launcher

Simple launcher script to start the FractalSemantics GUI application.
Provides easy command-line access and dependency checking.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pandas
        import plotly
        import streamlit
        print("✅ All GUI dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install GUI dependencies:")
        print("pip install -r gui_requirements.txt")
        return False

def install_dependencies():
    """Install GUI dependencies."""
    print("📦 Installing GUI dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "gui_requirements.txt"
        ])
        print("✅ GUI dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def launch_streamlit_app():
    """Launch the Streamlit application."""
    gui_app_path = Path(__file__).parent / "gui_app.py"

    if not gui_app_path.exists():
        print(f"❌ GUI application not found at {gui_app_path}")
        return False

    print("🚀 Launching FractalSemantics GUI...")
    print(f"📍 Application: {gui_app_path}")
    print("🌐 Streamlit will open in your default browser")
    print("💡 Use Ctrl+C to stop the server")
    print("-" * 60)

    try:
        # Launch Streamlit with the GUI app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(gui_app_path)
        ])
        return True
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped")
        return True
    except Exception as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False

def main():
    """Main launcher function."""
    print("🔬 FractalSemantics GUI Launcher")
    print("=" * 40)

    # Check if dependencies are installed
    if not check_dependencies():
        print("\n🔧 Would you like to install the missing dependencies? (y/n)")
        choice = input().lower().strip()

        if choice in ['y', 'yes']:
            if not install_dependencies():
                sys.exit(1)
        else:
            print("Please install dependencies manually and try again.")
            sys.exit(1)

    # Launch the application
    if not launch_streamlit_app():
        sys.exit(1)

if __name__ == "__main__":
    main()
