#!/usr/bin/env python3
"""
Python Development Workflow for Cline

A comprehensive workflow for Python projects that includes:
- Code quality checks (ruff, black, mypy)
- Testing with pytest
- Security scanning
- Documentation generation
- Dependency management

Usage:
  /python-dev-workflow          - Run full development workflow
  /python-dev-workflow setup    - Setup development environment
  /python-dev-workflow check    - Run quality checks only
  /python-dev-workflow docs     - Generate documentation
  /python-dev-workflow deps     - Check dependencies
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class PythonDevWorkflow:
    """Comprehensive Python development workflow."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.venv_path = self.project_root / "venv"
        self.python_executable = self._get_python_executable()
        self.config = self._load_config()

    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable."""
        if self.venv_path.exists():
            return str(self.venv_path / "bin" / "python")
        return sys.executable

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .cline-workflow-config.json."""
        config_path = self.project_root / ".cline-workflow-config.json"

        default_config = {
            "dev_requirements": [
                "black>=22.0.0",
                "ruff>=0.1.0",
                "mypy>=1.0.0",
                "pytest>=7.0.0",
                "pytest-asyncio>=0.21.0",
                "pre-commit>=3.0.0",
                "safety>=2.0.0",
                "bandit>=1.7.0"
            ],
            "tool_versions": {},
            "skip_steps": [],
            "virtual_env_path": "venv",
            "test_patterns": ["tests/**/*.py", "**/test_*.py"]
        }

        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                # Merge with defaults
                return {**default_config, **user_config}
            except:
                pass

        return default_config

    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=False,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(f"   ✅ {result.stdout.strip()}")
            if result.stderr:
                print(f"   ⚠️  {result.stderr.strip()}")

            return result.returncode == 0

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def setup_environment(self) -> bool:
        """Set up the development environment."""
        print("🐍 Setting up Python development environment...")

        # Create virtual environment if it doesn't exist
        if not self.venv_path.exists():
            success = self._run_command(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                "Creating virtual environment"
            )
            if not success:
                return False

        # Upgrade pip
        success = self._run_command(
            [self.python_executable, "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrading pip"
        )

        # Install development dependencies
        dev_requirements = self.config.get("dev_requirements", [])

        for req in dev_requirements:
            success = self._run_command(
                [self.python_executable, "-m", "pip", "install", req],
                f"Installing {req}"
            )
            if not success:
                return False

        return True

    def code_quality_check(self) -> bool:
        """Run code quality checks."""
        print("🔍 Running code quality checks...")

        if "code_quality" in self.config.get("skip_steps", []):
            print("   ⏭️  Skipping code quality checks (configured to skip)")
            return True

        checks = [
            (["ruff", "check", "."], "Ruff linting"),
            (["black", "--check", "."], "Black formatting check"),
            (["mypy", "."], "MyPy type checking")
        ]

        all_passed = True
        for cmd, description in checks:
            success = self._run_command(cmd, description)
            if not success:
                all_passed = False

        return all_passed

    def security_scan(self) -> bool:
        """Run security scans."""
        print("🔒 Running security scans...")

        if "security" in self.config.get("skip_steps", []):
            print("   ⏭️  Skipping security scans (configured to skip)")
            return True

        scans = [
            (["safety", "check"], "Safety dependency security check"),
            (["bandit", "-r", "."], "Bandit security linting")
        ]

        all_passed = True
        for cmd, description in scans:
            success = self._run_command(cmd, description)
            if not success:
                all_passed = False

        return all_passed

    def run_tests(self) -> bool:
        """Run the test suite."""
        print("🧪 Running test suite...")

        if "testing" in self.config.get("skip_steps", []):
            print("   ⏭️  Skipping tests (configured to skip)")
            return True

        test_commands = [
            (["pytest", "-v"], "Running pytest"),
            (["pytest", "--cov", "."], "Running pytest with coverage")
        ]

        all_passed = True
        for cmd, description in test_commands:
            success = self._run_command(cmd, description)
            if not success:
                all_passed = False

        return all_passed

    def generate_docs(self) -> bool:
        """Generate documentation."""
        print("📚 Generating documentation...")

        if "documentation" in self.config.get("skip_steps", []):
            print("   ⏭️  Skipping documentation generation (configured to skip)")
            return True

        # Check if Sphinx is available
        success = self._run_command(
            [self.python_executable, "-c", "import sphinx; print('Sphinx available')"],
            "Checking Sphinx availability"
        )

        if success:
            # Try to build docs
            docs_dirs = ["docs", "doc", "documentation"]
            for docs_dir in docs_dirs:
                docs_path = self.project_root / docs_dir
                if docs_path.exists():
                    success = self._run_command(
                        [self.python_executable, "-m", "sphinx", "-b", "html", str(docs_path), str(docs_path / "_build" / "html")],
                        f"Building documentation in {docs_dir}"
                    )
                    if success:
                        return True

        print("   ⚠️  No documentation found or Sphinx not available")
        return True

    def check_dependencies(self) -> bool:
        """Check for outdated dependencies."""
        print("📦 Checking dependencies...")

        if "dependency_check" in self.config.get("skip_steps", []):
            print("   ⏭️  Skipping dependency check (configured to skip)")
            return True

        success = self._run_command(
            [self.python_executable, "-m", "pip", "list", "--outdated"],
            "Checking for outdated packages"
        )

        return success

    def run_setup(self) -> bool:
        """Run setup only."""
        print("🚀 Running Python Development Workflow Setup")
        print("=" * 50)

        return self.setup_environment()

    def run_check(self) -> bool:
        """Run quality checks only."""
        print("🚀 Running Python Development Workflow Quality Checks")
        print("=" * 50)

        steps = [
            (self.code_quality_check, "Code Quality"),
            (self.security_scan, "Security Scan"),
            (self.run_tests, "Testing")
        ]

        all_passed = True
        for step_func, step_name in steps:
            print(f"\n📋 {step_name}")
            print("-" * 30)

            success = step_func()
            if not success:
                print(f"❌ {step_name} failed")
                all_passed = False
            else:
                print(f"✅ {step_name} passed")

        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All quality checks completed successfully!")
        else:
            print("⚠️  Some quality checks failed. Please review the output above.")

        return all_passed

    def run_docs(self) -> bool:
        """Generate documentation only."""
        print("🚀 Running Python Development Workflow Documentation")
        print("=" * 50)

        return self.generate_docs()

    def run_deps(self) -> bool:
        """Check dependencies only."""
        print("🚀 Running Python Development Workflow Dependency Check")
        print("=" * 50)

        return self.check_dependencies()

    def run_full_workflow(self) -> bool:
        """Run the complete development workflow."""
        print("🚀 Starting Python Development Workflow")
        print("=" * 50)

        steps = [
            (self.setup_environment, "Environment Setup"),
            (self.code_quality_check, "Code Quality"),
            (self.security_scan, "Security Scan"),
            (self.run_tests, "Testing"),
            (self.generate_docs, "Documentation"),
            (self.check_dependencies, "Dependency Check")
        ]

        all_passed = True
        for step_func, step_name in steps:
            print(f"\n📋 {step_name}")
            print("-" * 30)

            success = step_func()
            if not success:
                print(f"❌ {step_name} failed")
                all_passed = False
            else:
                print(f"✅ {step_name} passed")

        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All workflow steps completed successfully!")
        else:
            print("⚠️  Some workflow steps failed. Please review the output above.")

        return all_passed


def main():
    """Main entry point for the workflow."""
    if len(sys.argv) < 2:
        # Default to full workflow
        workflow = PythonDevWorkflow()
        success = workflow.run_full_workflow()
        sys.exit(0 if success else 1)

    command = sys.argv[1].lower()

    if command == "setup":
        workflow = PythonDevWorkflow()
        success = workflow.run_setup()
        sys.exit(0 if success else 1)
    elif command == "check":
        workflow = PythonDevWorkflow()
        success = workflow.run_check()
        sys.exit(0 if success else 1)
    elif command == "docs":
        workflow = PythonDevWorkflow()
        success = workflow.run_docs()
        sys.exit(0 if success else 1)
    elif command == "deps":
        workflow = PythonDevWorkflow()
        success = workflow.run_deps()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  /python-dev-workflow          - Run full development workflow")
        print("  /python-dev-workflow setup    - Setup development environment")
        print("  /python-dev-workflow check    - Run quality checks only")
        print("  /python-dev-workflow docs     - Generate documentation")
        print("  /python-dev-workflow deps     - Check dependencies")
        sys.exit(1)


if __name__ == "__main__":
    main()
