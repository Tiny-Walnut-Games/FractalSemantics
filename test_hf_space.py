#!/usr/bin/env python3
"""
Test script for FractalSemantics Hugging Face Space

This script validates that the Hugging Face Space application can be imported
and that all dependencies are working correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test Gradio import
        import importlib.util
        if importlib.util.find_spec("gradio"):
            print("✓ Gradio available")
        else:
            print("✗ Gradio not available")
        
        # Test matplotlib
        import importlib.util
        if importlib.util.find_spec("matplotlib"):
            print("✓ Matplotlib available")
        else:
            print("✗ Matplotlib not available")

        # Test numpy
        if importlib.util.find_spec("numpy"):
            print("✓ NumPy available")
        else:
            print("✗ NumPy not available")

        # Test pandas
        if importlib.util.find_spec("pandas"):
            print("✓ Pandas available")
        else:
            print("✗ Pandas not available")

        # Test FractalSemantics modules
        try:
            if importlib.util.find_spec("fractalsemantics.fractalsemantics_experiments"):
                print("✓ FractalSemantics experiments available")
            else:
                print("✗ FractalSemantics experiments not available")
        except Exception:
            print("✗ FractalSemantics experiments check failed")

        try:
            import importlib.util
            if importlib.util.find_spec("fractalsemantics.config"):
                print("✓ FractalSemantics config available")
            else:
                print("✗ FractalSemantics config not available")
        except Exception:
            print("✗ FractalSemantics config check failed")

        # Test app module
        try:
            if importlib.util.find_spec("app"):
                print("✓ App module available")
            else:
                print("✗ App module not available")
        except Exception:
            print("✗ App module check failed")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_experiment_info():
    """Test that experiment information is properly defined."""
    print("\nTesting experiment information...")
    
    try:
        import app
        
        # Check that EXPERIMENT_INFO is defined
        if hasattr(app, 'EXPERIMENT_INFO'):
            experiment_count = len(app.EXPERIMENT_INFO)
            print(f"✓ Found {experiment_count} experiments defined")
            
            # Check a few key experiments
            key_experiments = ['exp01_geometric_collision', 'exp02_retrieval_efficiency', 'exp03_coordinate_entropy']
            for exp in key_experiments:
                if exp in app.EXPERIMENT_INFO:
                    info = app.EXPERIMENT_INFO[exp]
                    if all(key in info for key in ['title', 'description', 'math_concept', 'educational_content']):
                        print(f"✓ {exp} has complete information")
                    else:
                        print(f"✗ {exp} missing required fields")
                        return False
                else:
                    print(f"✗ {exp} not found in EXPERIMENT_INFO")
                    return False
            
            return True
        else:
            print("✗ EXPERIMENT_INFO not found in app module")
            return False
            
    except Exception as e:
        print(f"✗ Error testing experiment info: {e}")
        return False

def test_state_management():
    """Test that state management classes work correctly."""
    print("\nTesting state management...")
    
    try:
        import app
        
        # Test ExperimentState class
        state = app.ExperimentState()
        if hasattr(state, 'is_running') and hasattr(state, 'results'):
            print("✓ ExperimentState class works correctly")
            
            # Test initial state
            if not state.is_running and state.results == {}:
                print("✓ Initial state is correct")
                return True
            else:
                print("✗ Initial state is incorrect")
                return False
        else:
            print("✗ ExperimentState class missing required attributes")
            return False
            
    except Exception as e:
        print(f"✗ Error testing state management: {e}")
        return False

def test_chart_functions():
    """Test that chart creation functions work."""
    print("\nTesting chart functions...")
    
    try:
        import app
        import matplotlib.pyplot as plt
        
        # Test create_progress_chart
        progress_data = [(0, 0), (1, 25), (2, 50), (3, 75), (4, 100)]
        fig = app.create_progress_chart("Test Experiment", progress_data)
        if fig is not None:
            print("✓ create_progress_chart works")
            plt.close(fig)
        else:
            print("✗ create_progress_chart returned None")
            return False
        
        # Test create_results_chart
        results = {"success": True, "results": {"exit_code": 0, "stdout": "Test output"}}
        fig = app.create_results_chart("Test Experiment", results)
        if fig is not None:
            print("✓ create_results_chart works")
            plt.close(fig)
        else:
            print("✗ create_results_chart returned None")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing chart functions: {e}")
        return False

def test_gradio_interface():
    """Test that Gradio interface can be created."""
    print("\nTesting Gradio interface creation...")
    
    try:
        import app
        
        # Test interface creation
        demo = app.create_gradio_interface()
        if demo is not None:
            print("✓ Gradio interface created successfully")
            return True
        else:
            print("✗ Gradio interface creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Error creating Gradio interface: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("FRACTALSEMANTICS HUGGING FACE SPACE - VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_experiment_info,
        test_state_management,
        test_chart_functions,
        test_gradio_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Hugging Face Space is ready for deployment.")
        print("\nNext steps:")
        print("1. Run: python setup_hf_space.py deploy")
        print("2. Upload files to Hugging Face Spaces")
        print("3. Configure environment variables")
        print("4. Deploy and test the live application")
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        print("\nCommon issues and solutions:")
        print("- Import errors: Install missing dependencies with pip")
        print("- Module not found: Ensure all FractalSemantics files are present")
        print("- Chart errors: Install matplotlib and related visualization libraries")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
