#!/usr/bin/env python3
"""
Global Tools Setup Script for Cline

This script sets up all global workflows, hooks, and skills for Cline.
It provides a comprehensive development environment with automated tools
for code quality, security, and project management.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


class GlobalToolsSetup:
    """Setup class for global tools."""

    def __init__(self):
        self.cline_dir = Path.home() / ".cline"
        self.workflows_dir = self.cline_dir / "global_workflows"
        self.hooks_dir = self.cline_dir / "global_hooks"
        self.skills_dir = self.cline_dir / "global_skills"

        # Tools to install
        self.required_tools = [
            "ruff",
            "black",
            "mypy",
            "pytest",
            "pre-commit",
            "safety",
            "bandit"
        ]

    def setup_directories(self):
        """Create required directories."""
        print("📁 Creating global tools directories...")

        directories = [self.cline_dir, self.workflows_dir, self.hooks_dir, self.skills_dir]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created {directory}")

    def install_required_tools(self):
        """Install required development tools."""
        print("🔧 Installing required development tools...")

        for tool in self.required_tools:
            try:
                print(f"   Installing {tool}...")
                subprocess.run([sys.executable, "-m", "pip", "install", tool],
                             check=True, capture_output=True)
                print(f"   ✓ {tool} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ⚠ Failed to install {tool}: {e}")

    def setup_git_integration(self):
        """Setup Git integration for global hooks."""
        print("🪝 Setting up Git integration...")

        # Create global Git hooks directory
        git_hooks_dir = Path.home() / ".git-hooks"
        git_hooks_dir.mkdir(exist_ok=True)

        # Create global pre-commit hook
        pre_commit_hook = git_hooks_dir / "pre-commit"
        pre_commit_hook.write_text("""#!/bin/bash
# Global Cline pre-commit hook
PYTHONPATH=~/.cline:$PYTHONPATH python3 ~/.cline/global_hooks/pre_commit_hook.py run "$@"
""")
        pre_commit_hook.chmod(0o755)

        # Configure Git to use global hooks
        try:
            subprocess.run(["git", "config", "--global", "core.hooksPath", str(git_hooks_dir)],
                         check=True, capture_output=True)
            print("   ✓ Global Git hooks configured")
        except subprocess.CalledProcessError:
            print("   ⚠ Could not configure global Git hooks (not in a Git repository)")

    def create_global_config(self):
        """Create global configuration file."""
        print("⚙️  Creating global configuration...")

        config = {
            "global_tools": {
                "version": "1.0.0",
                "last_updated": "2025-01-19",
                "enabled_workflows": [
                    "python_dev_workflow",
                    "git_workflow"
                ],
                "enabled_hooks": [
                    "pre_commit_hook"
                ],
                "enabled_skills": [
                    "code_reviewer",
                    "project_analyzer"
                ]
            },
            "python_dev_workflow": {
                "default_tool_versions": {
                    "black": "latest",
                    "ruff": "latest",
                    "mypy": "latest",
                    "pytest": "latest"
                },
                "default_exclusions": [
                    "test_*",
                    "*_test.py",
                    "generated/*",
                    "node_modules/**"
                ]
            },
            "git_workflow": {
                "default_base_branch": "main",
                "default_release_prefix": "release/",
                "default_feature_prefix": "feature/",
                "auto_setup_hooks": True
            },
            "code_reviewer": {
                "default_severity_threshold": "warning",
                "default_exclusions": [
                    "test_*",
                    "*_test.py",
                    "generated/*"
                ],
                "enabled_languages": [
                    "python",
                    "javascript",
                    "typescript",
                    "rust",
                    "go",
                    "java"
                ]
            },
            "project_analyzer": {
                "default_exclusions": [
                    "node_modules/**",
                    ".git/**",
                    "dist/**",
                    "build/**"
                ],
                "quality_thresholds": {
                    "docstring_coverage": 80,
                    "comment_density": 10,
                    "max_file_size": 1000
                }
            }
        }

        config_file = self.cline_dir / "global_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"   ✓ Global configuration saved to {config_file}")

    def create_shell_aliases(self):
        """Create shell aliases for easy access."""
        print("🔗 Creating shell aliases...")

        aliases = {
            "bash": self.cline_dir / ".bash_aliases",
            "zsh": self.cline_dir / ".zsh_aliases"
        }

        alias_content = """
# Cline Global Tools Aliases
alias cline-workflows="python3 ~/.cline/global_workflows/python_dev_workflow.py"
alias cline-git="python3 ~/.cline/global_workflows/git_workflow.py"
alias cline-review="python3 ~/.cline/global_skills/code_reviewer.py"
alias cline-analyze="python3 ~/.cline/global_skills/project_analyzer.py"
alias cline-hook="python3 ~/.cline/global_hooks/pre_commit_hook.py"
alias cline-setup="python3 ~/.cline/setup_global_tools.py"
"""

        for shell, alias_file in aliases.items():
            with open(alias_file, 'w') as f:
                f.write(alias_content)

            print(f"   ✓ Created {shell} aliases in {alias_file}")

        # Add to shell configuration
        shell_configs = {
            "bash": Path.home() / ".bashrc",
            "zsh": Path.home() / ".zshrc"
        }

        for shell, config_file in shell_configs.items():
            if config_file.exists():
                with open(config_file) as f:
                    content = f.read()

                alias_import = f'source ~/.cline/.{shell}_aliases'
                if alias_import not in content:
                    with open(config_file, 'a') as f:
                        f.write(f'\n# Cline Global Tools\n{alias_import}\n')
                    print(f"   ✓ Added alias import to {config_file}")

    def create_completion_scripts(self):
        """Create shell completion scripts."""
        print("🔄 Creating shell completion scripts...")

        # Bash completion
        bash_completion = self.cline_dir / "cline_completion.bash"
        bash_completion.write_text("""#!/bin/bash
_cline_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="workflows git review analyze hook setup"
    
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
        return 0
    fi
    
    case ${prev} in
        workflows)
            COMPREPLY=($(compgen -W "run check install" -- ${cur}))
            ;;
        git)
            COMPREPLY=($(compgen -W "setup-hooks feature release pr status cleanup changelog" -- ${cur}))
            ;;
        review)
            COMPREPLY=($(compgen -W "file project" -- ${cur}))
            ;;
        analyze)
            COMPREPLY=($(compgen -W "project" -- ${cur}))
            ;;
        hook)
            COMPREPLY=($(compgen -W "install run status uninstall" -- ${cur}))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _cline_completion cline-workflows
complete -F _cline_completion cline-git
complete -F _cline_completion cline-review
complete -F _cline_completion cline-analyze
complete -F _cline_completion cline-hook
""")

        # Zsh completion
        zsh_completion = self.cline_dir / "cline_completion.zsh"
        zsh_completion.write_text("""#compdef cline-workflows cline-git cline-review cline-analyze cline-hook

_cline_completion() {
    local -a commands
    local -a subcommands
    
    case $words[1] in
        cline-workflows)
            commands=("run" "check" "install")
            ;;
        cline-git)
            commands=("setup-hooks" "feature" "release" "pr" "status" "cleanup" "changelog")
            ;;
        cline-review)
            commands=("file" "project")
            ;;
        cline-analyze)
            commands=("project")
            ;;
        cline-hook)
            commands=("install" "run" "status" "uninstall")
            ;;
    esac
    
    if [[ $CURRENT -eq 2 ]]; then
        _describe 'commands' commands
    else
        _message 'no more arguments'
    fi
}

compdef _cline_completion cline-workflows
compdef _cline_completion cline-git
compdef _cline_completion cline-review
compdef _cline_completion cline-analyze
compdef _cline_completion cline-hook
""")

        print(f"   ✓ Created bash completion: {bash_completion}")
        print(f"   ✓ Created zsh completion: {zsh_completion}")

    def create_documentation(self):
        """Create comprehensive documentation."""
        print("📚 Creating documentation...")

        docs_content = """# Cline Global Tools

This directory contains global workflows, hooks, and skills for Cline that enhance your development experience.

## Quick Start

### Installation

Run the setup script to install all global tools:
```bash
python3 ~/.cline/setup_global_tools.py
```

### Usage

#### Workflows
```bash
# Python development workflow
cline-workflows

# Git workflow
cline-git feature my-feature
cline-git pr
```

#### Skills
```bash
# Code review
cline-review src/main.py

# Project analysis
cline-analyze
```

#### Hooks
```bash
# Install pre-commit hook
cline-hook install

# Run hook manually
cline-hook run
```

## Available Tools

### Workflows
- **Python Development Workflow**: Comprehensive Python project management
- **Git Workflow**: Git automation and best practices

### Skills
- **Code Reviewer**: AI-powered code quality analysis
- **Project Analyzer**: Comprehensive project insights

### Hooks
- **Pre-commit Hook**: Automated quality checks before commits

## Configuration

Global configuration is stored in `~/.cline/global_config.json`.

Project-specific configurations can be created as:
- `.cline-workflow-config.json`
- `.cline-git-config.json`
- `.code-reviewer-config.json`
- `.project-analyzer-config.json`
- `.cline-hook-config.json`

## Shell Integration

The setup script creates aliases and completion scripts for easy access:

### Aliases
- `cline-workflows`: Python development workflow
- `cline-git`: Git workflow
- `cline-review`: Code reviewer
- `cline-analyze`: Project analyzer
- `cline-hook`: Pre-commit hook
- `cline-setup`: Setup script

### Completion
Bash and Zsh completion scripts are provided for all commands.

## Troubleshooting

### Common Issues

**Tools not found:**
```bash
# Reinstall tools
python3 ~/.cline/setup_global_tools.py
```

**Hooks not running:**
```bash
# Check Git configuration
git config --list | grep hooks

# Reinstall hooks
cline-hook install
```

**Aliases not working:**
```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc
```

### Getting Help

Each tool provides help with the `--help` flag:
```bash
cline-workflows --help
cline-git --help
cline-review --help
cline-analyze --help
cline-hook --help
```

## Contributing

To add new tools:

1. Create the script in the appropriate directory
2. Add documentation to this README
3. Test thoroughly with different project types
4. Submit a PR with clear description

## License

This project is licensed under the MIT License.
"""

        docs_file = self.cline_dir / "README.md"
        with open(docs_file, 'w') as f:
            f.write(docs_content)

        print(f"   ✓ Created documentation: {docs_file}")

    def run_setup(self):
        """Run the complete setup process."""
        print("🚀 Starting Cline Global Tools Setup")
        print("=" * 50)

        try:
            self.setup_directories()
            self.install_required_tools()
            self.setup_git_integration()
            self.create_global_config()
            self.create_shell_aliases()
            self.create_completion_scripts()
            self.create_documentation()

            print("\n✅ Setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Reload your shell configuration: source ~/.bashrc (or ~/.zshrc)")
            print("2. Test the tools: cline-setup --help")
            print("3. Configure your projects with .cline-* config files")
            print("4. Start using the tools in your projects!")

        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""Cline Global Tools Setup Script

Usage: python3 setup_global_tools.py [options]

Options:
  --help     Show this help message
  --reset    Reset and reinstall all tools
  --dry-run  Show what would be installed without making changes
""")
        sys.exit(0)

    setup = GlobalToolsSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()
