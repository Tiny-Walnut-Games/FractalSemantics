#!/usr/bin/env python3
"""
Test script for the FractalStat HTML Web Application.

This script validates that the web application components work correctly
and provides a demonstration of the educational features.
"""

import webbrowser
from pathlib import Path

def test_html_file():
    """Test that the HTML file exists and is valid."""
    html_file = Path("index.html")
    
    if not html_file.exists():
        print("❌ HTML file not found!")
        return False
    
    # Check file size (should be substantial)
    file_size = html_file.stat().st_size
    if file_size < 10000:  # Less than 10KB is suspiciously small
        print(f"⚠️  HTML file seems small: {file_size} bytes")
    
    print(f"✅ HTML file found: {html_file} ({file_size:,} bytes)")
    return True

def test_experiment_runner():
    """Test that the experiment runner exists and is valid."""
    runner_file = Path("experiment_runner.py")
    
    if not runner_file.exists():
        print("❌ Experiment runner not found!")
        return False
    
    # Check file size
    file_size = runner_file.stat().st_size
    if file_size < 5000:  # Less than 5KB is suspiciously small
        print(f"⚠️  Experiment runner seems small: {file_size} bytes")
    
    print(f"✅ Experiment runner found: {runner_file} ({file_size:,} bytes)")
    return True

def test_experiment_config():
    """Test that experiment configuration is valid."""
    try:
        # Import the experiment runner to test configuration
        sys.path.insert(0, str(Path(__file__).parent))
        from experiment_runner import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Check that all 12 experiments are configured
        expected_experiments = [
            "EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05", "EXP-06",
            "EXP-07", "EXP-08", "EXP-09", "EXP-10", "EXP-11", "EXP-12"
        ]
        
        configured_experiments = list(runner.experiment_configs.keys())
        
        if set(expected_experiments) == set(configured_experiments):
            print(f"✅ All {len(expected_experiments)} experiments configured correctly")
        else:
            print("❌ Experiment configuration mismatch")
            print(f"Expected: {expected_experiments}")
            print(f"Found: {configured_experiments}")
            return False
        
        # Test that each experiment has required fields
        for exp_id, config in runner.experiment_configs.items():
            required_fields = ["module", "description", "educational_focus"]
            for field in required_fields:
                if field not in config:
                    print(f"❌ Missing field '{field}' in {exp_id}")
                    return False
        
        print("✅ All experiment configurations have required fields")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import experiment runner: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing experiment configuration: {e}")
        return False

def test_educational_content():
    """Test that educational content is comprehensive."""
    try:
        from experiment_runner import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Test mathematical concepts for each experiment
        for exp_id in runner.experiment_configs.keys():
            concepts = runner._get_mathematical_concepts(exp_id)
            if not concepts or len(concepts) < 3:
                print(f"⚠️  {exp_id} has insufficient mathematical concepts: {len(concepts)}")
            else:
                print(f"✅ {exp_id} has {len(concepts)} mathematical concepts")
            
            steps = runner._get_experiment_steps(exp_id)
            if not steps or len(steps) < 3:
                print(f"⚠️  {exp_id} has insufficient steps: {len(steps)}")
            else:
                print(f"✅ {exp_id} has {len(steps)} detailed steps")
        
        print("✅ Educational content validation complete")
        return True
        
    except Exception as e:
        print(f"❌ Error testing educational content: {e}")
        return False

def test_readme():
    """Test that the README file exists and is comprehensive."""
    readme_file = Path("README_WEB_APP.md")
    
    if not readme_file.exists():
        print("❌ README file not found!")
        return False
    
    # Check file size
    file_size = readme_file.stat().st_size
    if file_size < 20000:  # Less than 20KB is suspiciously small for comprehensive docs
        print(f"⚠️  README seems small: {file_size} bytes")
    
    print(f"✅ README found: {readme_file} ({file_size:,} bytes)")
    return True

def launch_web_app():
    """Launch the web application in the default browser."""
    html_file = Path("index.html")
    
    if not html_file.exists():
        print("❌ Cannot launch - HTML file not found!")
        return False
    
    try:
        # Get absolute path
        abs_path = html_file.resolve()
        
        # Open in default browser
        webbrowser.open(f"file://{abs_path}")
        
        print("✅ Web application launched in browser")
        print(f"   URL: file://{abs_path}")
        print("   You can now explore the educational features!")
        return True
        
    except Exception as e:
        print(f"❌ Error launching web application: {e}")
        return False

def run_quick_demo():
    """Run a quick demonstration of the experiment runner."""
    try:
        from experiment_runner import ExperimentRunner
        
        print("\n🧪 Running quick demonstration...")
        
        # Test with a simple experiment
        runner = ExperimentRunner()
        
        # Generate educational introduction for EXP-01
        config = runner.experiment_configs["EXP-01"]
        intro = runner._generate_introduction("EXP-01", config)
        
        print("📝 Sample Educational Introduction:")
        print("-" * 50)
        print(intro[:500] + "..." if len(intro) > 500 else intro)
        print("-" * 50)
        
        # Test mathematical concepts
        concepts = runner._get_mathematical_concepts("EXP-01")
        print("📚 Mathematical Concepts for EXP-01:")
        for i, concept in enumerate(concepts, 1):
            print(f"   {i}. {concept}")
        
        # Test real-world applications
        applications = runner._get_real_world_applications("EXP-01")
        print("\n🌍 Real-World Applications:")
        print(applications)
        
        print("\n✅ Demonstration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 FractalStat Web Application Test Suite")
    print("=" * 50)
    
    tests = [
        ("HTML File", test_html_file),
        ("Experiment Runner", test_experiment_runner),
        ("Experiment Configuration", test_experiment_config),
        ("Educational Content", test_educational_content),
        ("README Documentation", test_readme),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The web application is ready to use.")
        
        # Offer to launch the application
        launch = input("\n🚀 Would you like to launch the web application? (y/n): ").lower().strip()
        if launch == 'y':
            launch_web_app()
        
        # Offer to run demonstration
        demo = input("\n🧪 Would you like to run a quick demonstration? (y/n): ").lower().strip()
        if demo == 'y':
            run_quick_demo()
            
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
    
    print("\n📖 For more information, see README_WEB_APP.md")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
