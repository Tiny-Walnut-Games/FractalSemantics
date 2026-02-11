#!/usr/bin/env python3
"""
Project Analyzer Skill for Cline

Comprehensive project analysis tool that provides insights into project structure,
dependencies, code quality metrics, and development patterns. Supports multiple
programming languages and project types.

Usage:
  /project-analyzer              - Analyze current project
  /project-analyzer <path>       - Analyze specific project
  /project-analyzer --metrics    - Show detailed metrics
  /project-analyzer --dependencies - Show dependency analysis
  /project-analyzer --structure  - Show project structure
  /project-analyzer --languages  - Show language analysis
"""

import ast
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProjectMetrics:
    """Project metrics and statistics."""
    total_files: int = 0
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    languages: Dict[str, int] = None
    file_types: Dict[str, int] = None
    largest_files: List[Tuple[str, int]] = None
    complexity_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.languages is None:
            object.__setattr__(self, 'languages', {})
        if self.file_types is None:
            object.__setattr__(self, 'file_types', {})
        if self.largest_files is None:
            object.__setattr__(self, 'largest_files', [])
        if self.complexity_metrics is None:
            object.__setattr__(self, 'complexity_metrics', {})


class ProjectAnalyzer:
    """Comprehensive project analysis tool."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.metrics = ProjectMetrics()
        self.config = self._load_config()
        self.file_analysis = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .cline-project-analyzer.json."""
        config_path = self.project_root / ".cline-project-analyzer.json"

        default_config = {
            "analysis_config": {
                "include_patterns": ["**/*"],
                "exclude_patterns": [
                    "node_modules/**",
                    ".git/**",
                    "dist/**",
                    "build/**",
                    "__pycache__/**",
                    "*.pyc",
                    "*.log"
                ],
                "max_file_size": 1024 * 1024,  # 1MB
                "analyze_complexity": True,
                "analyze_dependencies": True,
                "analyze_structure": True,
                "language_detection": {
                    "python": [".py"],
                    "javascript": [".js", ".jsx", ".ts", ".tsx"],
                    "rust": [".rs"],
                    "go": [".go"],
                    "java": [".java", ".kt"],
                    "c_cpp": [".c", ".cpp", ".h", ".hpp"],
                    "html": [".html", ".htm"],
                    "css": [".css", ".scss", ".sass"],
                    "config": [".json", ".yaml", ".yml", ".toml", ".xml", ".ini"]
                }
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

    def analyze_project(self) -> Dict[str, Any]:
        """Perform comprehensive project analysis."""
        print("🔍 Analyzing project structure...")

        # Collect basic metrics
        self._collect_file_metrics()

        # Analyze project structure
        structure = self._analyze_structure()

        # Analyze dependencies
        dependencies = self._analyze_dependencies()

        # Analyze languages
        languages = self._analyze_languages()

        # Generate report
        report = {
            "project_path": str(self.project_root),
            "metrics": asdict(self.metrics),
            "structure": structure,
            "dependencies": dependencies,
            "languages": languages,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _collect_file_metrics(self):
        """Collect basic file and line metrics."""
        print("📊 Collecting file metrics...")

        include_patterns = self.config.get("analysis_config", {}).get("include_patterns", ["**/*"])
        exclude_patterns = self.config.get("analysis_config", {}).get("exclude_patterns", [])
        max_file_size = self.config.get("analysis_config", {}).get("max_file_size", 1024 * 1024)

        total_files = 0
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0

        file_types = defaultdict(int)
        languages = defaultdict(int)
        largest_files = []

        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    # Check exclusions
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break

                    if should_exclude:
                        continue

                    # Check file size
                    if file_path.stat().st_size > max_file_size:
                        continue

                    total_files += 1
                    file_types[file_path.suffix] += 1

                    # Analyze file content
                    try:
                        with open(file_path, encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.split('\n')

                            file_lines = len(lines)
                            total_lines += file_lines

                            file_code_lines = 0
                            file_comment_lines = 0
                            file_blank_lines = 0

                            for line in lines:
                                stripped = line.strip()
                                if not stripped:
                                    file_blank_lines += 1
                                elif self._is_comment_line(stripped, file_path.suffix):
                                    file_comment_lines += 1
                                else:
                                    file_code_lines += 1

                            code_lines += file_code_lines
                            comment_lines += file_comment_lines
                            blank_lines += file_blank_lines

                            # Track largest files
                            largest_files.append((str(file_path), file_lines))

                            # Language detection
                            lang = self._detect_language(file_path)
                            if lang:
                                languages[lang] += 1
                                self.file_analysis[str(file_path)] = {
                                    "lines": file_lines,
                                    "code_lines": file_code_lines,
                                    "comment_lines": file_comment_lines,
                                    "blank_lines": file_blank_lines,
                                    "language": lang
                                }

                    except Exception:
                        # Skip files that can't be read
                        continue

        # Sort largest files
        largest_files.sort(key=lambda x: x[1], reverse=True)
        largest_files = largest_files[:10]  # Top 10

        self.metrics.total_files = total_files
        self.metrics.total_lines = total_lines
        self.metrics.code_lines = code_lines
        self.metrics.comment_lines = comment_lines
        self.metrics.blank_lines = blank_lines
        self.metrics.file_types = dict(file_types)
        self.metrics.languages = dict(languages)
        self.metrics.largest_files = largest_files

    def _is_comment_line(self, line: str, file_extension: str) -> bool:
        """Check if a line is a comment based on file type."""
        if not line:
            return False

        # Python, JavaScript, Java, C++ comments
        if file_extension in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.kt', '.c', '.cpp', '.h', '.hpp']:
            return line.startswith('#') or line.startswith('//') or line.startswith('/*') or line.startswith('*')

        # HTML comments
        if file_extension in ['.html', '.htm']:
            return line.startswith('<!--')

        # YAML comments
        if file_extension in ['.yaml', '.yml']:
            return line.startswith('#')

        return False

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language based on file extension."""
        language_detection = self.config.get("analysis_config", {}).get("language_detection", {})

        for lang, extensions in language_detection.items():
            if file_path.suffix in extensions:
                return lang

        return None

    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze project directory structure."""
        print("🏗️  Analyzing project structure...")

        structure = {
            "directories": [],
            "file_distribution": {},
            "depth_analysis": {},
            "patterns": []
        }

        # Collect directory structure
        directories = set()
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                # Add parent directories
                current = file_path.parent
                while current != self.project_root and current != current.parent:
                    directories.add(str(current.relative_to(self.project_root)))
                    current = current.parent

        structure["directories"] = sorted(list(directories))

        # Analyze file distribution
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                parent = str(file_path.parent.relative_to(self.project_root))
                if parent not in structure["file_distribution"]:
                    structure["file_distribution"][parent] = 0
                structure["file_distribution"][parent] += 1

        # Analyze depth
        max_depth = 0
        depth_counts = defaultdict(int)

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                depth = len(file_path.relative_to(self.project_root).parts) - 1
                max_depth = max(max_depth, depth)
                depth_counts[depth] += 1

        structure["depth_analysis"] = {
            "max_depth": max_depth,
            "depth_distribution": dict(depth_counts)
        }

        # Detect common patterns
        patterns = []

        # Check for src/ directory
        if (self.project_root / "src").exists():
            patterns.append("src-based structure")

        # Check for test directories
        test_dirs = list(self.project_root.glob("**/test*"))
        if test_dirs:
            patterns.append(f"test directories: {len(test_dirs)}")

        # Check for configuration files
        config_files = ["package.json", "requirements.txt", "Cargo.toml", "go.mod", "pom.xml"]
        found_configs = [f for f in config_files if (self.project_root / f).exists()]
        if found_configs:
            patterns.append(f"config files: {', '.join(found_configs)}")

        structure["patterns"] = patterns

        return structure

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        print("📦 Analyzing dependencies...")

        dependencies = {
            "python": self._analyze_python_dependencies(),
            "javascript": self._analyze_javascript_dependencies(),
            "rust": self._analyze_rust_dependencies(),
            "go": self._analyze_go_dependencies(),
            "java": self._analyze_java_dependencies()
        }

        return dependencies

    def _analyze_python_dependencies(self) -> Dict[str, Any]:
        """Analyze Python dependencies."""
        python_deps = {
            "requirements_files": [],
            "import_analysis": {},
            "virtual_envs": []
        }

        # Find requirements files
        for req_file in ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py"]:
            if (self.project_root / req_file).exists():
                python_deps["requirements_files"].append(req_file)

        # Find virtual environments
        for venv_dir in ["venv", ".venv", "env"]:
            if (self.project_root / venv_dir).exists():
                python_deps["virtual_envs"].append(venv_dir)

        # Analyze imports in Python files
        import_counts = defaultdict(int)

        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                import_counts[alias.name.split('.')[0]] += 1
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                import_counts[node.module.split('.')[0]] += 1
            except:
                continue

        python_deps["import_analysis"] = dict(import_counts)

        return python_deps

    def _analyze_javascript_dependencies(self) -> Dict[str, Any]:
        """Analyze JavaScript dependencies."""
        js_deps = {
            "package_files": [],
            "dependencies": {},
            "dev_dependencies": {}
        }

        # Find package.json
        if (self.project_root / "package.json").exists():
            js_deps["package_files"].append("package.json")

            try:
                with open(self.project_root / "package.json") as f:
                    package_data = json.load(f)

                    if "dependencies" in package_data:
                        js_deps["dependencies"] = package_data["dependencies"]

                    if "devDependencies" in package_data:
                        js_deps["dev_dependencies"] = package_data["devDependencies"]
            except:
                pass

        return js_deps

    def _analyze_rust_dependencies(self) -> Dict[str, Any]:
        """Analyze Rust dependencies."""
        rust_deps = {
            "cargo_files": [],
            "dependencies": {}
        }

        # Find Cargo.toml
        if (self.project_root / "Cargo.toml").exists():
            rust_deps["cargo_files"].append("Cargo.toml")

            try:
                with open(self.project_root / "Cargo.toml") as f:
                    content = f.read()
                    # Simple parsing for dependencies section
                    in_deps = False
                    for line in content.split('\n'):
                        line = line.strip()
                        if line == "[dependencies]":
                            in_deps = True
                            continue
                        elif line.startswith('[') and in_deps:
                            break
                        elif in_deps and '=' in line:
                            dep_name = line.split('=')[0].strip()
                            rust_deps["dependencies"][dep_name] = "unknown"
            except:
                pass

        return rust_deps

    def _analyze_go_dependencies(self) -> Dict[str, Any]:
        """Analyze Go dependencies."""
        go_deps = {
            "go_mod_files": [],
            "dependencies": {}
        }

        # Find go.mod
        if (self.project_root / "go.mod").exists():
            go_deps["go_mod_files"].append("go.mod")

            try:
                with open(self.project_root / "go.mod") as f:
                    content = f.read()
                    # Simple parsing for require section
                    in_require = False
                    for line in content.split('\n'):
                        line = line.strip()
                        if line == "require (":
                            in_require = True
                            continue
                        elif line == ")" and in_require:
                            break
                        elif in_require and ' ' in line:
                            dep_name = line.split()[0]
                            go_deps["dependencies"][dep_name] = "unknown"
            except:
                pass

        return go_deps

    def _analyze_java_dependencies(self) -> Dict[str, Any]:
        """Analyze Java dependencies."""
        java_deps = {
            "build_files": [],
            "dependencies": {}
        }

        # Find Maven/Gradle files
        build_files = ["pom.xml", "build.gradle", "build.gradle.kts"]
        for build_file in build_files:
            if (self.project_root / build_file).exists():
                java_deps["build_files"].append(build_file)

        return java_deps

    def _analyze_languages(self) -> Dict[str, Any]:
        """Analyze language usage and patterns."""
        print("🌍 Analyzing languages...")

        language_analysis = {
            "primary_language": None,
            "language_breakdown": self.metrics.languages,
            "file_patterns": {},
            "code_quality_indicators": {}
        }

        # Determine primary language
        if self.metrics.languages:
            primary_lang = max(self.metrics.languages.items(), key=lambda x: x[1])
            language_analysis["primary_language"] = primary_lang[0]

        # Analyze file patterns
        for file_path, analysis in self.file_analysis.items():
            lang = analysis.get("language")
            if lang:
                if lang not in language_analysis["file_patterns"]:
                    language_analysis["file_patterns"][lang] = {
                        "total_files": 0,
                        "avg_file_size": 0,
                        "complexity_indicators": []
                    }

                lang_data = language_analysis["file_patterns"][lang]
                lang_data["total_files"] += 1
                lang_data["avg_file_size"] += analysis["lines"]

        # Calculate averages
        for lang, data in language_analysis["file_patterns"].items():
            if data["total_files"] > 0:
                data["avg_file_size"] /= data["total_files"]

        # Code quality indicators
        quality_indicators = {
            "comment_ratio": 0,
            "complexity_warnings": [],
            "file_size_warnings": []
        }

        if self.metrics.total_lines > 0:
            quality_indicators["comment_ratio"] = self.metrics.comment_lines / self.metrics.total_lines

        # Check for large files
        for file_path, size in self.metrics.largest_files:
            if size > 1000:
                quality_indicators["file_size_warnings"].append(f"{file_path}: {size} lines")

        language_analysis["code_quality_indicators"] = quality_indicators

        return language_analysis

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Comment ratio recommendations
        if self.metrics.total_lines > 0:
            comment_ratio = self.metrics.comment_lines / self.metrics.total_lines
            if comment_ratio < 0.1:
                recommendations.append("Consider adding more comments to improve code documentation")
            elif comment_ratio > 0.3:
                recommendations.append("High comment ratio - ensure comments are valuable and up-to-date")

        # File size recommendations
        for file_path, size in self.metrics.largest_files[:3]:
            if size > 500:
                recommendations.append(f"Consider refactoring large file: {file_path} ({size} lines)")

        # Language-specific recommendations
        if "python" in self.metrics.languages:
            if self.metrics.languages["python"] > 10:
                recommendations.append("Consider using type hints for better code maintainability")

        if "javascript" in self.metrics.languages:
            recommendations.append("Consider using a linter (ESLint) for consistent code style")

        # Structure recommendations
        if self.metrics.total_files > 100:
            recommendations.append("Consider organizing files into more specific directories")

        if not any(d.startswith("test") for d in os.listdir(self.project_root) if os.path.isdir(d)):
            recommendations.append("Consider adding a test directory for better project organization")

        return recommendations

    def show_metrics(self) -> bool:
        """Show detailed metrics."""
        print("📊 Project Metrics")
        print("=" * 50)

        print(f"📁 Total files: {self.metrics.total_files}")
        print(f"📝 Total lines: {self.metrics.total_lines}")
        print(f"💻 Code lines: {self.metrics.code_lines}")
        print(f"💬 Comment lines: {self.metrics.comment_lines}")
        print(f"📄 Blank lines: {self.metrics.blank_lines}")

        if self.metrics.total_lines > 0:
            print(f"📊 Comment ratio: {(self.metrics.comment_lines/self.metrics.total_lines)*100:.1f}%")

        print("\n🌍 Languages:")
        for lang, count in self.metrics.languages.items():
            print(f"  {lang}: {count} files")

        print("\n📁 File types:")
        for ext, count in self.metrics.file_types.items():
            print(f"  {ext}: {count} files")

        print("\n📏 Largest files:")
        for file_path, size in self.metrics.largest_files[:5]:
            print(f"  {file_path}: {size} lines")

        return True

    def show_dependencies(self) -> bool:
        """Show dependency analysis."""
        print("📦 Dependency Analysis")
        print("=" * 50)

        deps = self._analyze_dependencies()

        for lang, analysis in deps.items():
            if analysis:
                print(f"\n{lang.upper()} Dependencies:")
                for key, value in analysis.items():
                    if isinstance(value, dict) and value:
                        print(f"  {key}: {len(value)} items")
                        for item, version in list(value.items())[:5]:
                            print(f"    {item}: {version}")
                    elif isinstance(value, list) and value:
                        print(f"  {key}: {', '.join(value)}")
                    elif value:
                        print(f"  {key}: {value}")

        return True

    def show_structure(self) -> bool:
        """Show project structure."""
        print("🏗️  Project Structure")
        print("=" * 50)

        structure = self._analyze_structure()

        print(f"📁 Directories: {len(structure['directories'])}")
        print(f"📊 Max depth: {structure['depth_analysis']['max_depth']}")

        print("\n📁 Top directories by file count:")
        sorted_dirs = sorted(structure["file_distribution"].items(), key=lambda x: x[1], reverse=True)
        for dir_path, count in sorted_dirs[:10]:
            print(f"  {dir_path}: {count} files")

        print("\n🔍 Patterns detected:")
        for pattern in structure["patterns"]:
            print(f"  • {pattern}")

        return True

    def show_languages(self) -> bool:
        """Show language analysis."""
        print("🌍 Language Analysis")
        print("=" * 50)

        languages = self._analyze_languages()

        print(f"Primary language: {languages['primary_language']}")

        print("\n📊 Language breakdown:")
        for lang, count in languages["language_breakdown"].items():
            print(f"  {lang}: {count} files")

        print("\n📈 Code quality indicators:")
        quality = languages["code_quality_indicators"]
        print(f"  Comment ratio: {quality['comment_ratio']*100:.1f}%")

        if quality["file_size_warnings"]:
            print("  Large files:")
            for warning in quality["file_size_warnings"][:5]:
                print(f"    {warning}")

        return True

    def show_help(self) -> bool:
        """Show available commands."""
        print("🔧 Project Analyzer Commands:")
        print()
        print("  (no args)                - Analyze current project")
        print("  <path>                   - Analyze specific project")
        print("  --metrics                - Show detailed metrics")
        print("  --dependencies           - Show dependency analysis")
        print("  --structure              - Show project structure")
        print("  --languages              - Show language analysis")
        print()
        print("Configuration file: .cline-project-analyzer.json")

        return True


def main():
    """Main entry point for the project analyzer."""
    if len(sys.argv) < 2:
        analyzer = ProjectAnalyzer()
        report = analyzer.analyze_project()

        # Print summary
        print("🔍 Project Analysis Complete")
        print("=" * 50)
        print(f"📁 Project: {report['project_path']}")
        print(f"📊 Files: {report['metrics']['total_files']}")
        print(f"📝 Lines: {report['metrics']['total_lines']}")
        print(f"🌍 Primary language: {report['languages']['primary_language']}")

        if report['recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['recommendations'][:5]:
                print(f"  • {rec}")

        return

    command = sys.argv[1].lower()
    project_path = sys.argv[2] if len(sys.argv) > 2 else None

    analyzer = ProjectAnalyzer(project_path)

    if command == "--metrics":
        analyzer._collect_file_metrics()
        analyzer.show_metrics()
    elif command == "--dependencies":
        analyzer.show_dependencies()
    elif command == "--structure":
        analyzer.show_structure()
    elif command == "--languages":
        analyzer.show_languages()
    elif command == "--help":
        analyzer.show_help()
    else:
        print("Usage:")
        print("  /project-analyzer              - Analyze current project")
        print("  /project-analyzer <path>       - Analyze specific project")
        print("  /project-analyzer --metrics    - Show detailed metrics")
        print("  /project-analyzer --dependencies - Show dependency analysis")
        print("  /project-analyzer --structure  - Show project structure")
        print("  /project-analyzer --languages  - Show language analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
