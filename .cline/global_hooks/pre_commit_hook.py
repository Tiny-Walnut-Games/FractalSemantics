#!/usr/bin/env python3
"""
Global Pre-Commit Hook for Cline

A comprehensive pre-commit hook that runs quality checks before allowing commits.
Can be used with any Git repository to enforce code quality standards.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class PreCommitHook:
    """Comprehensive pre-commit hook for code quality."""

    def __init__(self):
        self.repo_root = self._get_repo_root()
        self.config = self._load_config()

    def _get_repo_root(self) -> Path:
        """Get the root directory of the Git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True
            )
            return Path(result.stdout.strip())
        except:
            return Path.cwd()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .cline-pre-commit.json or use defaults."""
        config_path = self.repo_root / ".cline-pre-commit.json"

        default_config = {
            "enabled_checks": [
                "linting",
                "formatting",
                "type_checking",
                "security",
                "tests"
            ],
            "linting": {
                "tools": ["ruff", "flake8"],
                "fail_on_warning": True
            },
            "formatting": {
                "tools": ["black", "isort"],
                "check_only": True
            },
            "type_checking": {
                "tools": ["mypy"],
                "strict": False
            },
            "security": {
                "tools": ["bandit", "safety"],
                "fail_on_warning": True
            },
            "tests": {
                "tools": ["pytest"],
                "run_on_changed_files": True
            },
            "file_patterns": {
                "python": ["*.py"],
                "javascript": ["*.js", "*.ts", "*.jsx", "*.tsx"],
                "rust": ["*.rs"],
                "go": ["*.go"]
            }
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

    def _get_staged_files(self) -> List[Path]:
        """Get list of staged files that will be committed."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True
            )

            files = []
            for file_path in result.stdout.strip().split('\n'):
                if file_path:
                    files.append(self.repo_root / file_path)

            return files
        except:
            return []

    def _get_changed_python_files(self) -> List[Path]:
        """Get list of changed Python files."""
        staged_files = self._get_staged_files()
        return [f for f in staged_files if f.suffix == '.py']

    def _run_command(self, cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
        """Run a command and return success status."""
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_root,
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

    def check_linting(self) -> bool:
        """Run linting checks on changed files."""
        print("🔍 Running linting checks...")

        python_files = self._get_changed_python_files()
        if not python_files:
            print("   ✅ No Python files to lint")
            return True

        tools = self.config.get("linting", {}).get("tools", [])
        all_passed = True

        for tool in tools:
            if tool == "ruff":
                success = self._run_command(
                    ["ruff", "check"] + [str(f) for f in python_files],
                    f"Running {tool} on {len(python_files)} files"
                )
            elif tool == "flake8":
                success = self._run_command(
                    ["flake8"] + [str(f) for f in python_files],
                    f"Running {tool} on {len(python_files)} files"
                )
            else:
                print(f"   ⚠️  Unknown linting tool: {tool}")
                success = True

            if not success and self.config.get("linting", {}).get("fail_on_warning", True):
                all_passed = False

        return all_passed

    def check_formatting(self) -> bool:
        """Check code formatting."""
        print("🎨 Checking code formatting...")

        python_files = self._get_changed_python_files()
        if not python_files:
            print("   ✅ No Python files to format")
            return True

        tools = self.config.get("formatting", {}).get("tools", [])
        check_only = self.config.get("formatting", {}).get("check_only", True)
        all_passed = True

        for tool in tools:
            if tool == "black":
                cmd = ["black", "--check"] if check_only else ["black"]
                success = self._run_command(
                    cmd + [str(f) for f in python_files],
                    f"Running {tool} on {len(python_files)} files"
                )
            elif tool == "isort":
                cmd = ["isort", "--check-only"] if check_only else ["isort"]
                success = self._run_command(
                    cmd + [str(f) for f in python_files],
                    f"Running {tool} on {len(python_files)} files"
                )
            else:
                print(f"   ⚠️  Unknown formatting tool: {tool}")
                success = True

            if not success:
                all_passed = False

        return all_passed

    def check_type_annotations(self) -> bool:
        """Run type checking."""
        print("🏷️  Checking type annotations...")

        python_files = self._get_changed_python_files()
        if not python_files:
            print("   ✅ No Python files to type check")
            return True

        tools = self.config.get("type_checking", {}).get("tools", [])
        strict = self.config.get("type_checking", {}).get("strict", False)
        all_passed = True

        for tool in tools:
            if tool == "mypy":
                cmd = ["mypy"]
                if strict:
                    cmd.append("--strict")
                success = self._run_command(
                    cmd + [str(f) for f in python_files],
                    f"Running {tool} on {len(python_files)} files"
                )
            else:
                print(f"   ⚠️  Unknown type checking tool: {tool}")
                success = True

            if not success:
                all_passed = False

        return all_passed

    def check_security(self) -> bool:
        """Run security checks."""
        print("🔒 Running security checks...")

        tools = self.config.get("security", {}).get("tools", [])
        all_passed = True

        for tool in tools:
            if tool == "bandit":
                python_files = self._get_changed_python_files()
                if python_files:
                    success = self._run_command(
                        ["bandit", "-r"] + [str(f) for f in python_files],
                        f"Running {tool} on {len(python_files)} files"
                    )
                else:
                    success = True
            elif tool == "safety":
                success = self._run_command(
                    ["safety", "check"],
                    f"Running {tool}"
                )
            else:
                print(f"   ⚠️  Unknown security tool: {tool}")
                success = True

            if not success and self.config.get("security", {}).get("fail_on_warning", True):
                all_passed = False

        return all_passed

    def run_tests(self) -> bool:
        """Run tests on changed files."""
        print("🧪 Running tests...")

        tools = self.config.get("tests", {}).get("tools", [])
        run_on_changed = self.config.get("tests", {}).get("run_on_changed_files", True)
        all_passed = True

        for tool in tools:
            if tool == "pytest":
                if run_on_changed:
                    python_files = self._get_changed_python_files()
                    if python_files:
                        # Try to find test files related to changed files
                        test_files = []
                        for py_file in python_files:
                            # Look for corresponding test files
                            test_file = py_file.parent / f"test_{py_file.name}"
                            if test_file.exists():
                                test_files.append(str(test_file))

                        if test_files:
                            success = self._run_command(
                                ["pytest", "-v"] + test_files,
                                f"Running {tool} on {len(test_files)} test files"
                            )
                        else:
                            success = True
                    else:
                        success = True
                else:
                    success = self._run_command(
                        ["pytest", "-v"],
                        f"Running {tool}"
                    )
            else:
                print(f"   ⚠️  Unknown test tool: {tool}")
                success = True

            if not success:
                all_passed = False

        return all_passed

    def check_commit_message(self) -> bool:
        """Check commit message format."""
        print("📝 Checking commit message...")

        # Get the commit message
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True
            )
            message = result.stdout.strip()

            if not message:
                print("   ⚠️  No commit message found")
                return True

            # Basic commit message checks
            lines = message.split('\n')
            first_line = lines[0]

            if len(first_line) > 72:
                print(f"   ⚠️  First line too long ({len(first_line)} chars, max 72)")
                return False

            if first_line.endswith('.'):
                print("   ⚠️  First line should not end with a period")
                return False

            print("   ✅ Commit message format looks good")
            return True

        except:
            print("   ⚠️  Could not check commit message")
            return True

    def run_all_checks(self) -> bool:
        """Run all enabled checks."""
        print("🚀 Running pre-commit checks")
        print("=" * 50)

        enabled_checks = self.config.get("enabled_checks", [])

        checks = {
            "linting": self.check_linting,
            "formatting": self.check_formatting,
            "type_checking": self.check_type_annotations,
            "security": self.check_security,
            "tests": self.run_tests,
            "commit_message": self.check_commit_message
        }

        all_passed = True
        for check_name in enabled_checks:
            if check_name in checks:
                print(f"\n📋 {check_name.replace('_', ' ').title()}")
                print("-" * 30)

                success = checks[check_name]()
                if not success:
                    print(f"❌ {check_name} failed")
                    all_passed = False
                else:
                    print(f"✅ {check_name} passed")
            else:
                print(f"⚠️  Unknown check: {check_name}")

        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All pre-commit checks passed!")
        else:
            print("⚠️  Some pre-commit checks failed. Please fix the issues before committing.")

        return all_passed


def main():
    """Main entry point for the pre-commit hook."""
    hook = PreCommitHook()
    success = hook.run_all_checks()

    if not success:
        print("\n💡 Tips:")
        print("  - Run 'black .' to auto-format your code")
        print("  - Run 'ruff check .' to see linting issues")
        print("  - Run 'mypy .' to check type annotations")
        print("  - Use 'git add .' to stage your fixes")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
