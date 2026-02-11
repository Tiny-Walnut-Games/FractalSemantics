#!/usr/bin/env python3
"""
Setup script for the Anthropic/Claude-style markdown-based system.

This script helps integrate the markdown-based workflows, hooks, and skills
with the existing Python tools and MCP server configuration.
"""

import json
import os
import shutil
import sys
from pathlib import Path


class MarkdownSystemSetup:
    """Setup and integration for the markdown-based system."""

    def __init__(self, project_root: str = ""):
        self.project_root = Path(project_root or os.getcwd())
        self.cline_dir = self.project_root / ".cline"
        self.home_cline_dir = Path.home() / ".cline"

    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.cline_dir / "workflows",
            self.cline_dir / "hooks",
            self.cline_dir / "skills",
            self.cline_dir / "config"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")

    def copy_existing_tools(self):
        """Copy existing Python tools to the new system."""
        source_dirs = [
            self.cline_dir / "global_workflows",
            self.cline_dir / "global_hooks",
            self.cline_dir / "global_skills"
        ]

        target_dirs = [
            self.cline_dir / "workflows",
            self.cline_dir / "hooks",
            self.cline_dir / "skills"
        ]

        for source, target in zip(source_dirs, target_dirs):
            if source.exists():
                for file in source.glob("*.py"):
                    target_file = target / file.name
                    shutil.copy2(file, target_file)
                    print(f"✅ Copied {file} to {target_file}")

    def create_symlinks(self):
        """Create symlinks for easy access."""
        # Create symlinks from old global_* directories to new directories
        old_dirs = ["global_workflows", "global_hooks", "global_skills"]
        new_dirs = ["workflows", "hooks", "skills"]

        for old_dir, new_dir in zip(old_dirs, new_dirs):
            old_path = self.cline_dir / old_dir
            new_path = self.cline_dir / new_dir

            if new_path.exists() and not old_path.exists():
                try:
                    old_path.symlink_to(new_path)
                    print(f"✅ Created symlink: {old_path} -> {new_path}")
                except OSError as e:
                    print(f"⚠️  Could not create symlink: {e}")

    def setup_mcp_server(self):
        """Setup MCP server configuration."""
        mcp_config = self.cline_dir / "mcp-server-config.json"

        if mcp_config.exists():
            print(f"✅ MCP server configuration already exists: {mcp_config}")
        else:
            print(f"⚠️  MCP server configuration not found. Please ensure {mcp_config} exists.")

    def create_integration_scripts(self):
        """Create integration scripts for the system."""

        # Create wrapper script for Python tools
        wrapper_script = self.cline_dir / "run_tool.py"
        wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper script for running Python tools from the markdown system.
"""
import sys
import subprocess
import os
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tool.py <tool_path> [args...]")
        sys.exit(1)
    
    tool_path = Path(sys.argv[1])
    args = sys.argv[2:]
    
    if not tool_path.exists():
        print(f"Error: Tool not found: {tool_path}")
        sys.exit(1)
    
    # Run the tool
    result = subprocess.run([sys.executable, str(tool_path)] + args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
'''

        with open(wrapper_script, 'w') as f:
            f.write(wrapper_content)

        # Make executable
        os.chmod(wrapper_script, 0o755)
        print(f"✅ Created wrapper script: {wrapper_script}")

    def create_example_configurations(self):
        """Create example configuration files."""

        configs = {
            "cline-workflow-config.json": {
                "python_dev_workflow": {
                    "dev_requirements": ["custom-package>=1.0.0"],
                    "tool_versions": {"black": "22.3.0"},
                    "skip_steps": ["documentation"]
                }
            },
            "cline-git-config.json": {
                "git_workflow": {
                    "base_branch": "main",
                    "pre_commit_hooks": ["ruff", "black", "mypy"]
                }
            },
            "cline-pre-commit.json": {
                "enabled_checks": ["linting", "formatting", "security"],
                "linting": {"tools": ["ruff", "flake8"], "fail_on_warning": True}
            },
            "cline-code-reviewer.json": {
                "reviewer_config": {
                    "enabled_categories": ["security", "performance", "quality"],
                    "severity_levels": {"security": "error", "performance": "warning"}
                }
            },
            "cline-project-analyzer.json": {
                "analyzer_config": {
                    "enabled_categories": ["complexity", "coverage", "architecture"],
                    "complexity_thresholds": {"function": 10, "class": 20}
                }
            }
        }

        for filename, config in configs.items():
            config_file = self.cline_dir / "config" / filename
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Created example config: {config_file}")

    def validate_setup(self):
        """Validate the setup and provide recommendations."""

        print("\n🔍 Validating setup...")

        # Check directories
        required_dirs = ["workflows", "hooks", "skills"]
        for dir_name in required_dirs:
            dir_path = self.cline_dir / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/ directory exists")
            else:
                print(f"❌ {dir_name}/ directory missing")

        # Check MCP config
        mcp_config = self.cline_dir / "mcp-server-config.json"
        if mcp_config.exists():
            print("✅ MCP server configuration exists")
            try:
                with open(mcp_config) as f:
                    config = json.load(f)
                print(f"✅ Found {len(config.get('presets', {}))} presets")
            except json.JSONDecodeError:
                print("❌ MCP configuration is invalid JSON")
        else:
            print("❌ MCP server configuration missing")

        # Check for markdown files
        markdown_files = list(self.cline_dir.glob("**/*.md"))
        if markdown_files:
            print(f"✅ Found {len(markdown_files)} markdown files")
        else:
            print("⚠️  No markdown files found")

        # Check for Python tools
        python_tools = list(self.cline_dir.glob("**/*.py"))
        if python_tools:
            print(f"✅ Found {len(python_tools)} Python tools")
        else:
            print("⚠️  No Python tools found")

    def print_usage_guide(self):
        """Print usage guide and next steps."""

        print("\n" + "="*60)
        print("🎉 Markdown System Setup Complete!")
        print("="*60)

        print("\n📋 Available Commands:")
        print("  /python-dev-workflow    - Python development automation")
        print("  /git-workflow          - Git workflow management")
        print("  /pre-commit-hook       - Quality checks before commits")
        print("  /code-reviewer <file>  - AI-powered code review")
        print("  /project-analyzer .    - Project health analysis")

        print("\n🔧 Configuration Files:")
        print("  .cline/config/cline-workflow-config.json")
        print("  .cline/config/cline-git-config.json")
        print("  .cline/config/cline-pre-commit.json")
        print("  .cline/config/cline-code-reviewer.json")
        print("  .cline/config/cline-project-analyzer.json")

        print("\n🚀 Next Steps:")
        print("  1. Review and customize configuration files")
        print("  2. Test commands: /python-dev-workflow")
        print("  3. Setup pre-commit hooks: /git-workflow setup-hooks")
        print("  4. Integrate with CI/CD pipelines")
        print("  5. Train custom models if needed")

        print("\n📚 Documentation:")
        print("  - See .cline/README.md for detailed usage")
        print("  - Check individual tool markdown files for options")
        print("  - Review MCP server configuration for customization")

        print("\n💡 Tips:")
        print("  - Use --debug flag for troubleshooting")
        print("  - Configure tool-specific settings for your project")
        print("  - Integrate with your IDE for better workflow")
        print("  - Monitor metrics and track improvements over time")

    def run_full_setup(self):
        """Run the complete setup process."""
        print("🚀 Setting up Anthropic/Claude-style Markdown System")
        print("="*60)

        self.setup_directories()
        self.copy_existing_tools()
        self.create_symlinks()
        self.setup_mcp_server()
        self.create_integration_scripts()
        self.create_example_configurations()
        self.validate_setup()
        self.print_usage_guide()


def main():
    """Main entry point for the setup script."""
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    setup = MarkdownSystemSetup(project_root)
    setup.run_full_setup()


if __name__ == "__main__":
    main()
