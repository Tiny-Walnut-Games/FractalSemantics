#!/usr/bin/env python3
"""
Hugging Face Space Setup Script

This script helps set up and validate the FractalStat Interactive Experiments
Hugging Face Space environment. It checks dependencies, validates configuration,
and provides helpful setup information.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any


class HFSpaceSetup:
    """Hugging Face Space setup and validation utility."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements_hf.txt"
        self.app_yaml = self.project_root / "app.yaml"
        self.app_py = self.project_root / "app.py"
        
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"
    
    def check_requirements(self) -> Tuple[bool, List[str], List[str]]:
        """Check if all required packages are installed."""
        if not self.requirements_file.exists():
            return False, [], ["requirements_hf.txt not found"]
        
        required_packages = []
        with open(self.requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before >=, ==, etc.)
                    package = line.split('>=')[0].split('==')[0].split('<=')[0].split('!=')[0].strip()
                    required_packages.append(package)
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    installed_packages.append(package)
                else:
                    missing_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        success = len(missing_packages) == 0
        return success, installed_packages, missing_packages
    
    def check_fractalstat_modules(self) -> Tuple[bool, List[str], List[str]]:
        """Check if FractalStat modules are available."""
        fractalstat_dir = self.project_root / "fractalstat"
        if not fractalstat_dir.exists():
            return False, [], ["fractalstat directory not found"]
        
        required_modules = [
            "fractalstat_experiments.py",
            "fractalstat_entity.py",
            "config/__init__.py"
        ]
        
        missing_modules = []
        available_modules = []
        
        for module in required_modules:
            module_path = fractalstat_dir / module
            if module_path.exists():
                available_modules.append(module)
            else:
                missing_modules.append(module)
        
        success = len(missing_modules) == 0
        return success, available_modules, missing_modules
    
    def validate_app_yaml(self) -> Tuple[bool, List[str]]:
        """Validate the app.yaml configuration file."""
        errors = []
        
        if not self.app_yaml.exists():
            return False, ["app.yaml not found"]
        
        try:
            import yaml
        except ImportError:
            return False, ["PyYAML not installed - cannot validate app.yaml"]
        
        try:
            with open(self.app_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['runtime', 'storage', 'resources', 'network']
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")
            
            # Validate runtime type
            if 'runtime' in config and 'type' in config['runtime']:
                if config['runtime']['type'] != 'python':
                    errors.append("Runtime type must be 'python'")
            
            # Validate requirements file
            if 'runtime' in config and 'requirements' in config['runtime']:
                req_file = config['runtime']['requirements']
                if not Path(req_file).exists():
                    errors.append(f"Requirements file '{req_file}' not found")
            
            # Validate app.py exists
            if not self.app_py.exists():
                errors.append("app.py not found")
            
        except Exception as e:
            errors.append(f"Error parsing app.yaml: {e}")
        
        return len(errors) == 0, errors
    
    def check_environment_variables(self) -> Dict[str, str]:
        """Check relevant environment variables."""
        env_vars = {
            'PYTHONUNBUFFERED': os.environ.get('PYTHONUNBUFFERED', 'Not set'),
            'GRADIO_SERVER_NAME': os.environ.get('GRADIO_SERVER_NAME', 'Not set'),
            'GRADIO_SERVER_PORT': os.environ.get('GRADIO_SERVER_PORT', 'Not set'),
            'FRACTALSTAT_ENV': os.environ.get('FRACTALSTAT_ENV', 'Not set')
        }
        return env_vars
    
    def run_dependency_install(self) -> Tuple[bool, str]:
        """Attempt to install missing dependencies."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                return True, "Dependencies installed successfully"
            else:
                return False, f"Failed to install dependencies: {result.stderr}"
        except Exception as e:
            return False, f"Error running pip install: {e}"
    
    def generate_deployment_info(self) -> Dict[str, Any]:
        """Generate deployment information and instructions."""
        python_ok, python_version = self.check_python_version()
        reqs_ok, installed, missing = self.check_requirements()
        modules_ok, available, unavailable = self.check_fractalstat_modules()
        yaml_ok, yaml_errors = self.validate_app_yaml()
        env_vars = self.check_environment_variables()
        
        return {
            "python_version": {
                "status": "OK" if python_ok else "ERROR",
                "version": python_version
            },
            "dependencies": {
                "status": "OK" if reqs_ok else "ERROR",
                "installed": installed,
                "missing": missing
            },
            "fractalstat_modules": {
                "status": "OK" if modules_ok else "ERROR",
                "available": available,
                "missing": unavailable
            },
            "configuration": {
                "status": "OK" if yaml_ok else "ERROR",
                "errors": yaml_errors
            },
            "environment": env_vars,
            "deployment_ready": all([python_ok, reqs_ok, modules_ok, yaml_ok])
        }
    
    def print_setup_summary(self):
        """Print a comprehensive setup summary."""
        info = self.generate_deployment_info()
        
        print("=" * 80)
        print("FRACTALSTAT HUGGING FACE SPACE - SETUP SUMMARY")
        print("=" * 80)
        
        # Python version
        print(f"\n🐍 Python Version: {info['python_version']['status']}")
        print(f"   {info['python_version']['version']}")
        
        # Dependencies
        print(f"\n📦 Dependencies: {info['dependencies']['status']}")
        if info['dependencies']['missing']:
            print(f"   Missing: {', '.join(info['dependencies']['missing'])}")
        print(f"   Installed: {len(info['dependencies']['installed'])} packages")
        
        # FractalStat modules
        print(f"\n🧩 FractalStat Modules: {info['fractalstat_modules']['status']}")
        if info['fractalstat_modules']['missing']:
            print(f"   Missing: {', '.join(info['fractalstat_modules']['missing'])}")
        print(f"   Available: {len(info['fractalstat_modules']['available'])} modules")
        
        # Configuration
        print(f"\n⚙️  Configuration: {info['configuration']['status']}")
        if info['configuration']['errors']:
            print(f"   Errors: {', '.join(info['configuration']['errors'])}")
        
        # Environment variables
        print("\n🌍 Environment Variables:")
        for var, value in info['environment'].items():
            print(f"   {var}: {value}")
        
        # Deployment readiness
        print(f"\n🚀 Deployment Status: {'READY' if info['deployment_ready'] else 'NOT READY'}")
        
        if not info['deployment_ready']:
            print("\n🔧 Recommended Actions:")
            if info['python_version']['status'] != "OK":
                print("   - Install Python 3.8 or higher")
            if info['dependencies']['status'] != "OK":
                print("   - Install missing dependencies: pip install -r requirements_hf.txt")
            if info['fractalstat_modules']['status'] != "OK":
                print("   - Ensure all FractalStat modules are present")
            if info['configuration']['status'] != "OK":
                print("   - Fix app.yaml configuration errors")
        
        print("\n" + "=" * 80)
        
        return info['deployment_ready']
    
    def create_deployment_guide(self) -> str:
        """Create a deployment guide file."""
        guide_content = """# FractalStat Hugging Face Space - Deployment Guide

## Quick Deployment

1. **Prerequisites Check**
   ```bash
   python setup_hf_space.py
   ```

2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements_hf.txt
   ```

3. **Create Hugging Face Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the Space type
   - Upload all project files

4. **Configure Environment**
   - Set `PYTHONUNBUFFERED=1`
   - Set `GRADIO_SERVER_NAME=0.0.0.0`
   - Set `GRADIO_SERVER_PORT=7860`

5. **Deploy**
   - Hugging Face will automatically build and deploy
   - Monitor the build logs for any issues

## File Structure

Required files for deployment:
```
fractalstat-hf-space/
├── app.py                    # Main application
├── requirements_hf.txt       # Dependencies
├── app.yaml                  # Hugging Face configuration
├── README_HF_SPACE.md        # Documentation
├── setup_hf_space.py         # Setup utility
└── fractalstat/              # Experiment modules
    ├── __init__.py
    ├── fractalstat_experiments.py
    ├── fractalstat_entity.py
    └── config/
        └── __init__.py
```

## Troubleshooting

### Import Errors
- Ensure all FractalStat modules are uploaded
- Check Python version compatibility (3.8+)
- Verify dependencies are installed

### Memory Issues
- Increase memory allocation in app.yaml
- Monitor experiment memory usage
- Consider running experiments sequentially

### Performance Issues
- Enable caching for repeated experiments
- Optimize visualization rendering
- Monitor CPU and memory usage

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the main README_HF_SPACE.md
- Report bugs via GitHub issues
"""
        
        guide_file = self.project_root / "DEPLOYMENT_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        return str(guide_file)


def main():
    """Main setup function."""
    setup = HFSpaceSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            # Just check the setup
            setup.print_setup_summary()
        
        elif command == "install":
            # Try to install dependencies
            success, message = setup.run_dependency_install()
            print(f"Installation: {'SUCCESS' if success else 'FAILED'}")
            print(message)
        
        elif command == "guide":
            # Create deployment guide
            guide_path = setup.create_deployment_guide()
            print(f"Deployment guide created at: {guide_path}")
        
        elif command == "deploy":
            # Full deployment check and setup
            ready = setup.print_setup_summary()
            if not ready:
                print("\nAttempting to fix issues...")
                success, message = setup.run_dependency_install()
                print(f"Dependency installation: {'SUCCESS' if success else 'FAILED'}")
                print(message)
            
            guide_path = setup.create_deployment_guide()
            print(f"\nDeployment guide created at: {guide_path}")
            print("You can now deploy to Hugging Face Spaces!")
        
        else:
            print("Usage: python setup_hf_space.py [check|install|guide|deploy]")
            print("  check  - Check setup status")
            print("  install - Install dependencies")
            print("  guide  - Create deployment guide")
            print("  deploy - Full deployment setup")
    
    else:
        # Default: check setup
        setup.print_setup_summary()


if __name__ == "__main__":
    main()
