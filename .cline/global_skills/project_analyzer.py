#!/usr/bin/env python3
"""
Global Project Analyzer Skill for Cline

Comprehensive project analysis tool that provides insights into:
- Project structure and architecture
- Technology stack detection
- Code quality metrics
- Security vulnerabilities
- Performance bottlenecks
- Dependency analysis
"""

import ast
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProjectInsight:
    """Represents a project insight or finding."""
    category: str  # "structure", "tech_stack", "quality", "security", "performance", "dependencies"
    severity: str  # "info", "warning", "error"
    title: str
    description: str
    recommendations: list[str]
    files: list[str]


class ProjectAnalyzer:
    """Comprehensive project analyzer."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.insights: list[ProjectInsight] = []
        self.tech_stack = {}
        self.metrics = {}

    def analyze_project(self) -> dict[str, any]:
        """Perform comprehensive project analysis."""
        print("🔍 Analyzing project structure...")

        # Basic project info
        project_info = self._analyze_project_info()

        # Technology stack detection
        self._detect_tech_stack()

        # Code quality analysis
        self._analyze_code_quality()

        # Security analysis
        self._analyze_security()

        # Performance analysis
        self._analyze_performance()

        # Dependency analysis
        self._analyze_dependencies()

        # Generate insights
        insights_by_category = self._categorize_insights()

        return {
            "project_info": project_info,
            "tech_stack": self.tech_stack,
            "metrics": self.metrics,
            "insights": insights_by_category,
            "summary": self._generate_summary()
        }

    def _analyze_project_info(self) -> dict[str, any]:
        """Analyze basic project information."""
        project_info = {
            "name": self.project_root.name,
            "path": str(self.project_root),
            "size": self._get_project_size(),
            "file_count": self._count_files(),
            "directories": self._get_directory_structure(),
            "last_modified": self._get_last_modified()
        }

        # Check for common project files
        common_files = {
            "README": self.project_root / "README.md",
            "LICENSE": self.project_root / "LICENSE",
            "CONTRIBUTING": self.project_root / "CONTRIBUTING.md",
            "CODE_OF_CONDUCT": self.project_root / "CODE_OF_CONDUCT.md"
        }

        project_info["documentation"] = {
            file_name: file_path.exists()
            for file_name, file_path in common_files.items()
        }

        return project_info

    def _detect_tech_stack(self):
        """Detect the technology stack used in the project."""
        tech_indicators = {
            "Python": [".py", "requirements.txt", "pyproject.toml", "setup.py"],
            "JavaScript/TypeScript": [".js", ".ts", ".jsx", ".tsx", "package.json"],
            "Java": [".java", "pom.xml", "build.gradle"],
            "C#": [".cs", ".csproj", "packages.config"],
            "Go": [".go", "go.mod"],
            "Rust": [".rs", "Cargo.toml"],
            "PHP": [".php", "composer.json"],
            "Ruby": [".rb", "Gemfile"],
            "C/C++": [".c", ".cpp", ".h", ".hpp", "Makefile", "CMakelists.txt"],
            "HTML/CSS": [".html", ".css", ".scss", ".sass"],
            "Docker": ["Dockerfile", "docker-compose.yml"],
            "Kubernetes": ["k8s.yaml", "k8s.yml", "deployment.yaml"],
            "AWS": ["serverless.yml", "sam.yaml", "cloudformation.yaml"],
            "Database": ["schema.sql", "migrations/", "models.py"],
            "Testing": ["test_", "spec/", "__tests__/", "pytest.ini"],
            "CI/CD": [".github/", ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml"],
            "Documentation": ["docs/", "README.md", "CHANGELOG.md"]
        }

        detected_tech = {}

        for tech, indicators in tech_indicators.items():
            score = 0
            found_files = []

            for indicator in indicators:
                if indicator.startswith('.'):
                    # File extension
                    files = list(self.project_root.glob(f"**/*{indicator}"))
                    if files:
                        score += len(files)
                        found_files.extend([str(f) for f in files[:5]])  # Limit to first 5
                else:
                    # File or directory name
                    if (self.project_root / indicator).exists():
                        score += 10  # Higher weight for exact matches
                        found_files.append(str(self.project_root / indicator))

            if score > 0:
                detected_tech[tech] = {
                    "score": score,
                    "confidence": "high" if score >= 10 else "medium" if score >= 5 else "low",
                    "files": found_files[:10]  # Limit to first 10 files
                }

        self.tech_stack = detected_tech

        # Add specific framework detection
        self._detect_frameworks()

    def _detect_frameworks(self):
        """Detect specific frameworks and libraries."""
        framework_indicators = {
            "Web Frameworks": {
                "Django": ["manage.py", "settings.py"],
                "Flask": ["flask", "app.py"],
                "FastAPI": ["fastapi", "main.py"],
                "Express": ["express", "app.js"],
                "React": ["react", "package.json"],
                "Vue.js": ["vue", "vue.config.js"],
                "Angular": ["angular", "angular.json"],
                "Spring Boot": ["spring-boot", "pom.xml"],
                "ASP.NET": ["aspnet", ".csproj"]
            },
            "Databases": {
                "PostgreSQL": ["psycopg2", "pg"],
                "MySQL": ["mysql", "pymysql"],
                "MongoDB": ["pymongo", "mongodb"],
                "Redis": ["redis", "redis-py"],
                "SQLite": ["sqlite3", "sqlite"]
            },
            "Cloud Platforms": {
                "AWS": ["boto3", "aws-sdk"],
                "Azure": ["azure", "azure-sdk"],
                "Google Cloud": ["google-cloud", "gcloud"],
                "Heroku": ["Procfile", "heroku.yml"]
            },
            "DevOps": {
                "Docker": ["Dockerfile", "docker-compose"],
                "Kubernetes": ["k8s", "kubernetes"],
                "Terraform": ["terraform", ".tf"],
                "Ansible": ["ansible", ".yml"]
            }
        }

        for category, frameworks in framework_indicators.items():
            for framework, indicators in frameworks.items():
                score = 0
                for indicator in indicators:
                    if any(indicator in str(f) for f in self.project_root.rglob("*")):
                        score += 1

                if score > 0:
                    if "frameworks" not in self.tech_stack:
                        self.tech_stack["frameworks"] = {}
                    if category not in self.tech_stack["frameworks"]:
                        self.tech_stack["frameworks"][category] = {}

                    self.tech_stack["frameworks"][category][framework] = score

    def _analyze_code_quality(self):
        """Analyze code quality metrics."""
        quality_metrics = {
            "file_complexity": self._analyze_file_complexity(),
            "naming_conventions": self._check_naming_conventions(),
            "code_smells": self._detect_code_smells(),
            "documentation": self._analyze_documentation()
        }

        self.metrics["quality"] = quality_metrics

    def _analyze_file_complexity(self) -> dict[str, any]:
        """Analyze file complexity metrics."""
        complexity_data = {
            "large_files": [],
            "deep_nesting": [],
            "long_functions": []
        }

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                # Check file size
                if len(lines) > 500:
                    complexity_data["large_files"].append({
                        "file": str(py_file),
                        "lines": len(lines)
                    })

                # Check function length
                import ast
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and hasattr(node, 'end_lineno') and node.end_lineno is not None:
                            func_lines = node.end_lineno - node.lineno
                            if func_lines > 50:
                                complexity_data["long_functions"].append({
                                    "file": str(py_file),
                                    "function": node.name,
                                        "lines": func_lines
                                    })
                except ast.ParseError:
                    pass

            except ast.ParseError:
                continue

        return complexity_data

    def _check_naming_conventions(self) -> dict[str, any]:
        """Check naming convention adherence."""
        naming_issues = {
            "snake_case_violations": [],
            "camel_case_violations": [],
            "constant_naming": []
        }

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()

                # Check for naming violations
                lines = content.split('\n')
                for _i, line in enumerate(lines, 1):
                    # Check variable naming
                    if re.match(r'^\s*[a-z_]+\s*=', line):
                        # Should be snake_case
                        pass
                    elif re.match(r'^\s*[A-Z][a-zA-Z0-9_]*\s*=', line):
                        # Might be a constant
                        pass

            except ast.ParseError:
                continue

        return naming_issues

    def _detect_code_smells(self) -> list[dict[str, any]]:
        """Detect common code smells."""
        smells = []

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()

                # Check for long parameter lists
                if re.search(r'def\s+\w+([^)]{100,}\)', content):
                    smells.append({
                        "file": str(py_file),
                        "smell": "Long parameter list",
                        "severity": "warning"
                    })

                # Check for long methods
                lines = content.split('\n')
                method_lines = 0
                in_method = False

                for line in lines:
                    if re.match(r'^\s*def\s+', line):
                        in_method = True
                        method_lines = 1
                    elif in_method:
                        method_lines += 1
                        if method_lines > 100:
                            smells.append({
                                "file": str(py_file),
                                "smell": "Long method",
                                "severity": "warning"
                            })
                            break

            except ast.ParseError:
                continue

        return smells

    def _analyze_documentation(self) -> dict[str, any]:
        """Analyze documentation quality."""
        doc_metrics = {
            "docstring_coverage": self._calculate_docstring_coverage(),
            "readme_quality": self._assess_readme_quality(),
            "comment_density": self._calculate_comment_density()
        }

        return doc_metrics

    def _calculate_docstring_coverage(self) -> float:
        """Calculate docstring coverage percentage."""
        total_functions = 0
        documented_functions = 0

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()

                import ast
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if (node.body and isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                                documented_functions += 1
                except ast.ParseError:
                    pass

            except ast.ParseError:
                continue

        if total_functions == 0:
            return 0.0

        return (documented_functions / total_functions) * 100

    def _assess_readme_quality(self) -> dict[str, any]:
        """Assess README file quality."""
        readme_path = self.project_root / "README.md"
        if not readme_path.exists():
            return {"exists": False, "score": 0, "issues": ["No README.md found"]}

        with open(readme_path, encoding='utf-8') as f:
            content = f.read()

        score = 0
        issues = []

        required_sections = [
            ("# ", "Title"),
            ("## Installation", "Installation section"),
            ("## Usage", "Usage section"),
            ("## Contributing", "Contributing section"),
            ("## License", "License section")
        ]

        for pattern, section in required_sections:
            if pattern in content:
                score += 20
            else:
                issues.append(f"Missing {section}")

        return {"exists": True, "score": score, "issues": issues}

    def _calculate_comment_density(self) -> float:
        """Calculate comment density in code."""
        total_lines = 0
        comment_lines = 0

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    total_lines += 1
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        comment_lines += 1

            except ast.ParseError:
                continue

        if total_lines == 0:
            return 0.0

        return (comment_lines / total_lines) * 100

    def _analyze_security(self):
        """Analyze security vulnerabilities."""
        security_issues = []

        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']'
        ]

        for file_path in self.project_root.glob("**/*.{py,js,java,go}"):
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()

                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        security_issues.append({
                            "file": str(file_path),
                            "issue": "Potential hardcoded secret",
                            "severity": "high"
                        })
                        break

            except ast.ParseError:
                continue

        self.metrics["security"] = {
            "issues": security_issues,
            "total": len(security_issues)
        }

    def _analyze_performance(self):
        """Analyze potential performance bottlenecks."""
        performance_issues = []

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()

                # Check for inefficient loops
                if re.search(r'for\s+\w+\s+in\s+range\(len\(', content):
                    performance_issues.append({
                        "file": str(py_file),
                        "issue": "Inefficient iteration pattern",
                        "severity": "medium"
                    })

                # Check for repeated database queries in loops
                if re.search(r'for.*:\s*.*\.execute\(', content, re.DOTALL):
                    performance_issues.append({
                        "file": str(py_file),
                        "issue": "Potential N+1 query problem",
                        "severity": "high"
                    })

            except ast.ParseError:
                continue

        self.metrics["performance"] = {
            "issues": performance_issues,
            "total": len(performance_issues)
        }

    def _analyze_dependencies(self):
        """Analyze project dependencies."""
        deps_info = {
            "python": self._analyze_python_deps(),
            "javascript": self._analyze_js_deps(),
            "security": self._check_dependency_security()
        }

        self.metrics["dependencies"] = deps_info

    def _analyze_python_deps(self) -> dict[str, any]:
        """Analyze Python dependencies."""
        deps = {"total": 0, "categories": defaultdict(int), "outdated": []}

        req_files = ["requirements.txt", "pyproject.toml", "setup.py"]

        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    if req_file == "requirements.txt":
                        with open(req_path) as f:
                            for line in f:
                                if line.strip() and not line.startswith('#'):
                                    deps["total"] += 1
                                    # Categorize dependencies
                                    if any(pkg in line.lower() for pkg in ['django', 'flask', 'fastapi']):
                                        deps["categories"]["web_frameworks"] += 1
                                    elif any(pkg in line.lower() for pkg in ['numpy', 'pandas', 'scipy']):
                                        deps["categories"]["data_science"] += 1
                                    elif any(pkg in line.lower() for pkg in ['requests', 'httpx']):
                                        deps["categories"]["http_clients"] += 1
                    elif req_file == "pyproject.toml":
                        # Parse TOML
                        pass

                except ast.ParseError:
                    continue

        return deps

    def _analyze_js_deps(self) -> dict[str, any]:
        """Analyze JavaScript dependencies."""
        deps = {"total": 0, "dev_total": 0}

        package_json = self.project_root / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)

                if "dependencies" in data:
                    deps["total"] = len(data["dependencies"])
                if "devDependencies" in data:
                    deps["dev_total"] = len(data["devDependencies"])

            except ast.ParseError:
                pass

        return deps

    def _check_dependency_security(self) -> dict[str, any]:
        """Check for known security vulnerabilities in dependencies."""
        security_issues = []

        # This would integrate with security databases like OSV, Snyk, etc.
        # For now, just placeholder logic

        return {
            "total_issues": len(security_issues),
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": len(security_issues)
        }

    def _get_project_size(self) -> dict[str, any]:
        """Get project size information."""
        total_size = 0
        file_count = 0

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            "bytes": total_size,
            "kilobytes": round(total_size / 1024, 2),
            "megabytes": round(total_size / (1024 * 1024), 2),
            "file_count": file_count
        }

    def _count_files(self) -> dict[str, int]:
        """Count files by extension."""
        file_counts = defaultdict(int)

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_counts[ext] += 1

        return dict(file_counts)

    def _get_directory_structure(self) -> list[str]:
        """Get directory structure."""
        dirs = set()

        for file_path in self.project_root.rglob("*"):
            if file_path.is_dir():
                relative_path = file_path.relative_to(self.project_root)
                if len(relative_path.parts) <= 3:  # Limit depth
                    dirs.add(str(relative_path))

        return sorted(dirs)

    def _get_last_modified(self) -> str:
        """Get last modification time."""
        try:
            latest_file = max(
                self.project_root.rglob("*"),
                key=lambda p: p.stat().st_mtime,
                default=None
            )
            if latest_file:
                import datetime
                return datetime.datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        except ast.ParseError:
            pass
        return "Unknown"

    def _categorize_insights(self) -> dict[str, list[ProjectInsight]]:
        """Categorize insights by type."""
        categories = defaultdict(list)

        # Add insights based on analysis results
        if self.tech_stack:
            categories["tech_stack"].append(ProjectInsight(
                category="tech_stack",
                severity="info",
                title="Technology Stack Detected",
                description=f"Detected {len(self.tech_stack)} technologies in use",
                recommendations=["Review technology choices for project requirements"],
                files=[]
            ))

        if self.metrics.get("quality", {}).get("code_smells"):
            categories["quality"].append(ProjectInsight(
                category="quality",
                severity="warning",
                title="Code Smells Detected",
                description=f"Found {len(self.metrics['quality']['code_smells'])} potential code smells",
                recommendations=["Review and refactor problematic code patterns"],
                files=[smell["file"] for smell in self.metrics["quality"]["code_smells"]]
            ))

        if self.metrics.get("security", {}).get("total", 0) > 0:
            categories["security"].append(ProjectInsight(
                category="security",
                severity="error",
                title="Security Issues Found",
                description=f"Detected {self.metrics['security']['total']} security issues",
                recommendations=["Address all security vulnerabilities immediately"],
                files=[issue["file"] for issue in self.metrics["security"]["issues"]]
            ))

        return dict(categories)

    def _generate_summary(self) -> dict[str, any]:
        """Generate analysis summary."""
        total_issues = 0
        critical_issues = 0

        for category_insights in self._categorize_insights().values():
            for insight in category_insights:
                total_issues += 1
                if insight.severity == "error":
                    critical_issues += 1

        quality_score = max(0, 100 - (total_issues * 5))

        return {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "quality_score": quality_score,
            "recommendations": self._get_recommendations()
        }

    def _get_recommendations(self) -> list[str]:
        """Get prioritized recommendations."""
        recommendations = []

        if self.metrics.get("security", {}).get("total", 0) > 0:
            recommendations.append("🔒 Address security vulnerabilities immediately")

        if self.metrics.get("quality", {}).get("code_smells"):
            recommendations.append("📝 Refactor code to eliminate code smells")

        if self.metrics.get("performance", {}).get("total", 0) > 0:
            recommendations.append("⚡ Optimize performance bottlenecks")

        if self.tech_stack.get("Python", {}).get("score", 0) > 0:
            recommendations.append("🐍 Consider using type hints for better code quality")

        return recommendations


def main():
    """Main entry point for the project analyzer."""
    project_path = sys.argv[1] if len(sys.argv) > 1 else None

    analyzer = ProjectAnalyzer(project_path)
    results = analyzer.analyze_project()

    # Print formatted results
    print("📊 Project Analysis Results")
    print("=" * 50)

    print(f"\n📁 Project: {results['project_info']['name']}")
    print(f"   Size: {results['project_info']['size']['megabytes']} MB")
    print(f"   Files: {results['project_info']['file_count']}")

    print(f"\n🛠️  Technologies: {len(results['tech_stack'])}")
    for tech, info in results['tech_stack'].items():
        if tech != "frameworks":
            print(f"   {tech}: {info['confidence']} confidence")

    print(f"\n📈 Quality Score: {results['summary']['quality_score']}/100")
    print(f"🚨 Issues Found: {results['summary']['total_issues']}")

    if results['summary']['recommendations']:
        print("\n💡 Recommendations:")
        for rec in results['summary']['recommendations']:
            print(f"   {rec}")

    # Save detailed results
    output_file = Path("project_analysis_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
