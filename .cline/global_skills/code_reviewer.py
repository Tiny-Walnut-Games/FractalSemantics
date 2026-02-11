#!/usr/bin/env python3
"""
Global Code Reviewer Skill for Cline

An AI-powered code reviewer that provides comprehensive feedback on code quality,
security, performance, and best practices. Can be used across any project.
"""

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReviewIssue:
    """Represents a code review issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "security", "performance", "style", "logic", "best_practice"
    line: int
    column: int
    message: str
    suggestion: Optional[str] = None


class CodeReviewer:
    """AI-powered code reviewer for multiple languages."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.issues: List[ReviewIssue] = []

    def review_file(self, file_path: Path) -> List[ReviewIssue]:
        """Review a single file and return list of issues."""
        self.issues = []

        if not file_path.exists():
            return self.issues

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except:
            return self.issues

        # Determine file type and run appropriate reviews
        file_extension = file_path.suffix.lower()

        if file_extension == '.py':
            self._review_python_file(file_path, content, lines)
        elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
            self._review_javascript_file(file_path, content, lines)
        elif file_extension == '.rs':
            self._review_rust_file(file_path, content, lines)
        elif file_extension == '.go':
            self._review_go_file(file_path, content, lines)
        elif file_extension in ['.java', '.kt']:
            self._review_java_file(file_path, content, lines)
        elif file_extension in ['.c', '.cpp', '.h', '.hpp']:
            self._review_c_cpp_file(file_path, content, lines)
        elif file_extension in ['.html', '.htm']:
            self._review_html_file(file_path, content, lines)
        elif file_extension in ['.css', '.scss', '.sass']:
            self._review_css_file(file_path, content, lines)
        elif file_extension in ['.json', '.yaml', '.yml', '.toml']:
            self._review_config_file(file_path, content, lines)

        return self.issues

    def _review_python_file(self, file_path: Path, content: str, lines: List[str]):
        """Review Python files for common issues."""

        # Parse AST for deeper analysis
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self._add_issue("error", "syntax", e.lineno, e.offset or 0,
                          f"Syntax error: {e.msg}")
            return

        # Check for common security issues
        self._check_python_security(content, lines)

        # Check for performance issues
        self._check_python_performance(content, lines)

        # Check for code style issues
        self._check_python_style(content, lines)

        # Check for best practices
        self._check_python_best_practices(content, lines)

        # Check imports
        self._check_python_imports(content, lines)

        # Check docstrings
        self._check_python_docstrings(tree, lines)

    def _check_python_security(self, content: str, lines: List[str]):
        """Check for security vulnerabilities in Python code."""

        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'private[_-]?key\s*=\s*["\'][^"\']{20,}["\']'
        ]

        for i, line in enumerate(lines, 1):
            for pattern in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_issue("error", "security", i, 0,
                                  "Potential hardcoded secret detected",
                                  "Use environment variables or secure configuration management")

        # Check for unsafe eval/exec
        unsafe_patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bcompile\s*\('
        ]

        for i, line in enumerate(lines, 1):
            for pattern in unsafe_patterns:
                if re.search(pattern, line):
                    self._add_issue("error", "security", i, 0,
                                  "Use of potentially unsafe function",
                                  "Avoid eval/exec/compile with user input")

        # Check for SQL injection patterns
        sql_patterns = [
            r'execute\s*\(\s*["\'][^"\']*%s',
            r'execute\s*\(\s*["\'][^"\']*{',
            r'cursor\.execute\s*\(\s*["\'][^"\']*%s'
        ]

        for i, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_issue("warning", "security", i, 0,
                                  "Potential SQL injection vulnerability",
                                  "Use parameterized queries instead")

    def _check_python_performance(self, content: str, lines: List[str]):
        """Check for performance issues in Python code."""

        # Check for inefficient loops
        for i, line in enumerate(lines, 1):
            if re.search(r'for\s+\w+\s+in\s+range\(len\(', line):
                self._add_issue("warning", "performance", i, 0,
                              "Inefficient iteration pattern",
                              "Use enumerate() or direct iteration instead")

            if re.search(r'\.append\(\)', line) in content:
                # Check for repeated appends in loops
                if i > 0 and i < len(lines) - 1:
                    if re.search(r'for\s+', lines[i-1]) or re.search(r'while\s+', lines[i-1]):
                        self._add_issue("info", "performance", i, 0,
                                      "Consider using list comprehension for better performance")

        # Check for unnecessary string concatenation in loops
        in_loop = False
        for i, line in enumerate(lines, 1):
            if re.search(r'for\s+.*\s+in\s+', line) or re.search(r'while\s+', line):
                in_loop = True
            elif line.strip() == '':
                continue
            elif in_loop and re.search(r'\+=\s*["\']', line):
                self._add_issue("info", "performance", i, 0,
                              "String concatenation in loop is inefficient",
                              "Use join() or f-strings instead")
                in_loop = False

    def _check_python_style(self, content: str, lines: List[str]):
        """Check for code style issues."""

        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > 88:  # Black default
                self._add_issue("warning", "style", i, 88,
                              f"Line too long ({len(line)} > 88 characters)")

        # Check for trailing whitespace
        for i, line in enumerate(lines, 1):
            if line.endswith(' ') or line.endswith('\t'):
                self._add_issue("info", "style", i, len(line),
                              "Trailing whitespace detected")

        # Check for multiple blank lines
        blank_count = 0
        for i, line in enumerate(lines, 1):
            if line.strip() == '':
                blank_count += 1
                if blank_count > 2:
                    self._add_issue("info", "style", i, 0,
                                  "Too many blank lines")
            else:
                blank_count = 0

    def _check_python_best_practices(self, content: str, lines: List[str]):
        """Check for Python best practices."""

        # Check for mutable default arguments
        for i, line in enumerate(lines, 1):
            if re.search(r'def\s+\w+.*=\s*(\[\]|\{\})', line):
                self._add_issue("error", "best_practice", i, 0,
                              "Mutable default argument detected",
                              "Use None and check inside function instead")

        # Check for bare except
        for i, line in enumerate(lines, 1):
            if re.search(r'except\s*:', line):
                self._add_issue("warning", "best_practice", i, 0,
                              "Bare except clause",
                              "Specify specific exception types")

        # Check for print statements in production code
        for i, line in enumerate(lines, 1):
            if re.search(r'\bprint\s*\(', line) and 'test' not in str(file_path).lower():
                self._add_issue("info", "best_practice", i, 0,
                              "Print statement in production code",
                              "Use logging instead of print")

    def _check_python_imports(self, content: str, lines: List[str]):
        """Check import statements."""

        imports = []
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*(import|from)\s+', line):
                imports.append((i, line.strip()))

        # Check import order (standard, third-party, local)
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        for line_num, import_line in imports:
            if import_line.startswith('from ') or import_line.startswith('import '):
                module_name = import_line.split()[1].split('.')[0]

                # Simple heuristic for categorization
                if module_name in ['os', 'sys', 'json', 're', 'datetime', 'collections']:
                    stdlib_imports.append((line_num, import_line))
                elif any(pkg in module_name.lower() for pkg in ['requests', 'numpy', 'pandas', 'django', 'flask']):
                    third_party_imports.append((line_num, import_line))
                else:
                    local_imports.append((line_num, import_line))

        # Check for unused imports (basic check)
        for line_num, import_line in imports:
            if 'import' in import_line:
                imported_name = import_line.split('import')[-1].strip().split(',')[0].strip()
                if imported_name not in content.replace(import_line, ''):
                    self._add_issue("info", "best_practice", line_num, 0,
                                  f"Potentially unused import: {imported_name}")

    def _check_python_docstrings(self, tree: ast.AST, lines: List[str]):
        """Check for missing docstrings."""

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)):
                    continue  # Has docstring
                else:
                    line_num = node.lineno
                    self._add_issue("info", "best_practice", line_num, 0,
                                  f"Missing docstring for {node.__class__.__name__.lower()}: {node.name}")

    def _review_javascript_file(self, file_path: Path, content: str, lines: List[str]):
        """Review JavaScript/TypeScript files."""
        # Similar structure to Python review but for JS/TS
        self._check_javascript_security(content, lines)
        self._check_javascript_performance(content, lines)
        self._check_javascript_style(content, lines)

    def _check_javascript_security(self, content: str, lines: List[str]):
        """Check for JavaScript security issues."""

        # Check for eval
        for i, line in enumerate(lines, 1):
            if re.search(r'\beval\s*\(', line):
                self._add_issue("error", "security", i, 0,
                              "Use of eval() is dangerous",
                              "Avoid eval() or sanitize input thoroughly")

        # Check for innerHTML with variables
        for i, line in enumerate(lines, 1):
            if re.search(r'\.innerHTML\s*=', line) and re.search(r'\$\{|\+.*\+', line):
                self._add_issue("warning", "security", i, 0,
                              "Potential XSS vulnerability",
                              "Use textContent or sanitize HTML")

    def _check_javascript_performance(self, content: str, lines: List[str]):
        """Check for JavaScript performance issues."""

        # Check for inefficient DOM queries in loops
        for i, line in enumerate(lines, 1):
            if re.search(r'document\.(getElement|querySelector)', line):
                # Check if this is in a loop (simplified check)
                if i > 0 and i < len(lines) - 1:
                    if re.search(r'(for|while)\s*\(.*\)\s*\{', lines[i-1] + lines[i]):
                        self._add_issue("warning", "performance", i, 0,
                                      "DOM query in loop is inefficient",
                                      "Cache DOM elements outside the loop")

    def _check_javascript_style(self, content: str, lines: List[str]):
        """Check JavaScript code style."""

        # Check semicolons
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.strip().startswith('//'):
                if not line.strip().endswith((';', '{', '}', ':', ',')):
                    if re.match(r'^\s*(var|let|const|function|if|for|while|return)', line):
                        self._add_issue("info", "style", i, len(line),
                                      "Missing semicolon")

    def _review_rust_file(self, file_path: Path, content: str, lines: List[str]):
        """Review Rust files."""
        # Basic Rust review patterns
        self._check_rust_borrowing(content, lines)
        self._check_rust_performance(content, lines)

    def _check_rust_borrowing(self, content: str, lines: List[str]):
        """Check for potential borrowing issues."""
        for i, line in enumerate(lines, 1):
            if re.search(r'\.clone\(\)', line):
                self._add_issue("info", "performance", i, 0,
                              "Consider avoiding clone() for performance")

    def _review_go_file(self, file_path: Path, content: str, lines: List[str]):
        """Review Go files."""
        # Basic Go review patterns
        self._check_go_error_handling(content, lines)
        self._check_go_performance(content, lines)

    def _check_go_error_handling(self, content: str, lines: List[str]):
        """Check for proper error handling."""
        for i, line in enumerate(lines, 1):
            if re.search(r'err\s*:=', line) and 'if err != nil' not in content:
                self._add_issue("warning", "best_practice", i, 0,
                              "Error not checked",
                              "Always check error return values")

    def _review_java_file(self, file_path: Path, content: str, lines: List[str]):
        """Review Java files."""
        # Basic Java review patterns
        self._check_java_naming(content, lines)
        self._check_java_performance(content, lines)

    def _review_c_cpp_file(self, file_path: Path, content: str, lines: List[str]):
        """Review C/C++ files."""
        # Basic C/C++ review patterns
        self._check_c_memory_management(content, lines)
        self._check_c_security(content, lines)

    def _review_html_file(self, file_path: Path, content: str, lines: List[str]):
        """Review HTML files."""
        # Basic HTML review patterns
        self._check_html_accessibility(content, lines)
        self._check_html_semantics(content, lines)

    def _review_css_file(self, file_path: Path, content: str, lines: List[str]):
        """Review CSS files."""
        # Basic CSS review patterns
        self._check_css_performance(content, lines)
        self._check_css_compatibility(content, lines)

    def _review_config_file(self, file_path: Path, content: str, lines: List[str]):
        """Review configuration files."""
        # Basic config review patterns
        self._check_config_secrets(content, lines)
        self._check_config_structure(content, lines)

    def _add_issue(self, severity: str, category: str, line: int, column: int, message: str, suggestion: Optional[str] = None):
        """Add an issue to the issues list."""
        self.issues.append(ReviewIssue(
            severity=severity,
            category=category,
            line=line,
            column=column,
            message=message,
            suggestion=suggestion
        ))

    def generate_report(self, file_path: Path, issues: List[ReviewIssue]) -> str:
        """Generate a formatted review report."""
        if not issues:
            return f"✅ {file_path.name}: No issues found!"

        report = f"🔍 Code Review Report: {file_path.name}\n"
        report += "=" * 50 + "\n\n"

        # Group issues by severity
        severity_groups = {"error": [], "warning": [], "info": []}
        for issue in issues:
            severity_groups[issue.severity].append(issue)

        for severity in ["error", "warning", "info"]:
            if severity_groups[severity]:
                report += f"🚨 {severity.upper()}S ({len(severity_groups[severity])})\n"
                report += "-" * 30 + "\n"

                for issue in severity_groups[severity]:
                    report += f"Line {issue.line}: {issue.message}\n"
                    if issue.suggestion:
                        report += f"   💡 Suggestion: {issue.suggestion}\n"
                    report += "\n"

        # Summary statistics
        report += "📊 Summary\n"
        report += "-" * 30 + "\n"
        report += f"Total issues: {len(issues)}\n"
        report += f"Errors: {len(severity_groups['error'])}\n"
        report += f"Warnings: {len(severity_groups['warning'])}\n"
        report += f"Info: {len(severity_groups['info'])}\n"

        return report

    def review_project(self, file_patterns: Optional[List[str]] = None) -> Dict[str, List[ReviewIssue]]:
        """Review entire project for issues."""
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.rs', '*.go', '*.java', '*.kt']

        all_issues = {}

        for pattern in file_patterns:
            for file_path in self.project_root.glob(f"**/{pattern}"):
                if file_path.is_file():
                    issues = self.review_file(file_path)
                    if issues:
                        all_issues[str(file_path)] = issues

        return all_issues


def main():
    """Main entry point for the code reviewer."""
    if len(sys.argv) < 2:
        print("Usage: python code_reviewer.py <file_path> [project_root]")
        print("Examples:")
        print("  python code_reviewer.py src/main.py")
        print("  python code_reviewer.py .  # Review entire project")
        sys.exit(1)

    target = Path(sys.argv[1])
    project_root = sys.argv[2] if len(sys.argv) > 2 else None

    reviewer = CodeReviewer(project_root)

    if target.is_file():
        # Review single file
        issues = reviewer.review_file(target)
        report = reviewer.generate_report(target, issues)
        print(report)
    elif target.is_dir():
        # Review entire project
        all_issues = reviewer.review_project()

        total_issues = sum(len(issues) for issues in all_issues.values())
        print("🔍 Project Review Complete")
        print(f"📁 Scanned: {target}")
        print(f"📊 Total issues found: {total_issues}")
        print()

        for file_path, issues in all_issues.items():
            print(reviewer.generate_report(Path(file_path), issues))
            print("=" * 80)
            print()
    else:
        print(f"❌ Error: {target} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
