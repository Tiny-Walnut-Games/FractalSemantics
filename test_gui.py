#!/usr/bin/env python3
"""
GUI Application Test Script

This script tests the FractalSemantics GUI application to ensure it works correctly
before deployment. It performs basic functionality tests and provides feedback.
"""

import sys
from pathlib import Path

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required imports work correctly."""
    print("🔍 Testing imports...")

    try:
        # Test core GUI dependencies
        import streamlit
        print("✅ Streamlit imported successfully")

        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")

        import pandas as pd
        print("✅ Pandas imported successfully")

        # Test FractalSemantics dependencies
        from fractalsemantics.experiment_runner import (
            ExperimentResult,
            ExperimentRunner,
        )
        print("✅ FractalSemantics experiment runner imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_experiment_runner():
    """Test that the experiment runner works correctly."""
    print("\n🧪 Testing experiment runner...")

    try:
        from fractalsemantics.experiment_runner import ExperimentRunner

        runner = ExperimentRunner()

        # Test that experiment configs are loaded
        if runner.experiment_configs:
            print(f"✅ Experiment configs loaded: {len(runner.experiment_configs)} experiments")

            # Test a quick experiment (EXP-01)
            print("⏱️  Running quick test of EXP-01...")
            import asyncio

            async def test_exp():
                result = await runner.run_experiment("EXP-01", quick_mode=True)
                return result

            result = asyncio.run(test_exp())

            if result:
                print(f"✅ EXP-01 test completed: {result.success}")
                print(f"   Duration: {result.duration:.2f}s")
                print(f"   Educational content: {len(result.educational_content)} sections")
                return True
            else:
                print("❌ Experiment result is None")
                return False

        else:
            print("❌ No experiment configs found")
            return False

    except Exception as e:
        print(f"❌ Experiment runner test failed: {e}")
        return False

def test_gui_components():
    """Test that GUI components can be imported and initialized."""
    print("\n🖥️  Testing GUI components...")

    try:
        # Import the GUI class
        from gui_app import FractalSemanticsGUI

        # Test initialization
        gui = FractalSemanticsGUI()
        print("✅ GUI class initialized successfully")

        # Test that runner is properly connected
        if hasattr(gui, 'runner') and gui.runner:
            print("✅ GUI connected to experiment runner")
        else:
            print("❌ GUI not properly connected to runner")
            return False

        # Test session state setup
        if hasattr(gui, 'setup_session_state'):
            gui.setup_session_state()
            print("✅ Session state setup completed")
        else:
            print("❌ Session state setup method not found")
            return False

        return True

    except Exception as e:
        print(f"❌ GUI component test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files are present."""
    print("\n📁 Testing file structure...")

    required_files = [
        "gui_app.py",
        "gui_requirements.txt",
        "launch_gui.py",
        "GUI_README.md"
    ]

    all_present = True

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} missing")
            all_present = False

    return all_present

def test_dependencies_file():
    """Test that the GUI requirements file is valid."""
    print("\n📦 Testing dependencies file...")

    try:
        requirements_file = Path("gui_requirements.txt")
        if not requirements_file.exists():
            print("❌ gui_requirements.txt not found")
            return False

        with open(requirements_file) as f:
            content = f.read()

        # Check for essential dependencies
        essential_deps = ["streamlit", "plotly", "pandas"]
        missing_deps = []

        for dep in essential_deps:
            if dep not in content:
                missing_deps.append(dep)

        if missing_deps:
            print(f"❌ Missing essential dependencies: {missing_deps}")
            return False
        else:
            print("✅ All essential dependencies found in requirements file")
            return True

    except Exception as e:
        print(f"❌ Requirements file test failed: {e}")
        return False

def performance_test():
    """Test basic performance characteristics."""
    print("\n⚡ Testing performance...")

    try:
        import time

        from fractalsemantics.experiment_runner import ExperimentRunner

        runner = ExperimentRunner()

        # Test quick experiment timing
        print("⏱️  Testing quick experiment timing...")
        start_time = time.time()

        import asyncio
        async def quick_test():
            result = await runner.run_experiment("EXP-01", quick_mode=True)
            return result

        asyncio.run(quick_test())
        duration = time.time() - start_time

        if duration < 30:  # Should complete in under 30 seconds
            print(f"✅ Quick experiment completed in {duration:.2f}s")
        else:
            print(f"⚠️  Experiment took longer than expected: {duration:.2f}s")

        # Test memory usage (basic check)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"✅ Current memory usage: {memory_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🔬 FractalSemantics GUI Test Suite")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies File", test_dependencies_file),
        ("Import Tests", test_imports),
        ("Experiment Runner", test_experiment_runner),
        ("GUI Components", test_gui_components),
        ("Performance", performance_test),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The GUI application is ready to use.")
        print("\n🚀 To launch the GUI:")
        print("   python launch_gui.py")
        print("   # or")
        print("   streamlit run gui_app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        print("\n🔧 Common fixes:")
        print("   - Install missing dependencies: pip install -r gui_requirements.txt")
        print("   - Install FractalSemantics: pip install -e .")
        print("   - Check Python version compatibility")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
