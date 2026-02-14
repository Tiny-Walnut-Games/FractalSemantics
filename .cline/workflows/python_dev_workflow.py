#!/usr/bin/env python3
"""
Global Python Development Workflow for Cline

A comprehensive workflow for Python projects that includes:
- Code quality checks (ruff, black, mypy)
- Testing with pytest
- Security scanning
- Documentation generation
- Dependency management
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, list


class PythonDevWorkflow:
    """Comprehensive Python development workflow."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.venv_path = self.project_root / "venv"
        self.python_executable = self._get_python_executable()

    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable."""
        if self.venv_path.exists():
            return str(self.venv_path / "bin" / "python")
        return sys.executable

    def _run_command(self, cmd: list[str], description: str) -> bool:
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
        dev_requirements = [
            "black>=22.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pre-commit>=3.0.0",
            "safety>=2.0.0",
            "bandit>=1.7.0"
        ]

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

        success = self._run_command(
            [self.python_executable, "-m", "pip", "list", "--outdated"],
            "Checking for outdated packages"
        )

        return success

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
    workflow = PythonDevWorkflow()
    success = workflow.run_full_workflow()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
