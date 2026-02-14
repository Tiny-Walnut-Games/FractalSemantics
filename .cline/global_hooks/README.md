# Global Hooks for Cline

This directory contains global hooks that can be used across all Cline workspaces to automate common development tasks and enforce code quality standards.

## Available Hooks

### Pre-commit Hook (`pre_commit_hook.py`)

A comprehensive pre-commit hook that runs quality checks before allowing commits. This hook ensures code quality, security, and consistency across your projects.

**Features:**

- **Code Quality Checks**: Runs Ruff, Black, and MyPy for Python projects
- **Security Scanning**: Uses Safety and Bandit for security analysis
- **File Validation**: Checks file sizes, naming conventions, and content
- **Git Integration**: Works with existing Git hooks and pre-commit framework
- **Configurable**: Respects project-specific configurations
- **Multi-language Support**: Handles Python, JavaScript, and other file types

**Usage:**

```bash
# Install as Git hook
python ~/.cline/global_hooks/pre_commit_hook.py install

# Run manually
python ~/.cline/global_hooks/pre_commit_hook.py run

# Check status
python ~/.cline/global_hooks/pre_commit_hook.py status

# Uninstall
python ~/.cline/global_hooks/pre_commit_hook.py uninstall
```

**Installation Options:**

- **Git Hook**: Installs as a Git pre-commit hook
- **Pre-commit Framework**: Integrates with pre-commit.com framework
- **Manual**: Can be run manually or integrated into CI/CD

## Configuration

### Hook Configuration

Create a `.cline-hook-config.json` file in your project root:

```json
{
  "pre_commit_hook": {
    "enabled_checks": [
      "code_quality",
      "security",
      "file_validation"
    ],
    "file_size_limit": 1048576,
    "exclude_patterns": [
      "test_*",
      "*_test.py",
      "generated/*",
      "node_modules/**"
    ],
    "tool_configs": {
      "ruff": {
        "config_file": "pyproject.toml"
      },
      "black": {
        "line_length": 88
      },
      "mypy": {
        "strict": true
      }
    }
  }
}
```

### Git Hook Integration

The hook can be installed as a Git pre-commit hook:

```bash
# Install hook
python ~/.cline/global_hooks/pre_commit_hook.py install

# Verify installation
git config --list | grep -i hook
```

### Pre-commit Framework Integration

For projects using the pre-commit framework, the hook can be added to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: cline-pre-commit-hook
        name: Cline Pre-commit Hook
        entry: python ~/.cline/global_hooks/pre_commit_hook.py run
        language: system
        pass_filenames: true
        stages: [commit]
```

## Check Types

### Code Quality Checks

- **Ruff**: Fast Python linter with comprehensive rule set
- **Black**: Code formatter for consistent Python style
- **MyPy**: Static type checker for Python
- **isort**: Import sorting and organization

### Security Checks

- **Safety**: Checks for known security vulnerabilities in dependencies
- **Bandit**: Security linter for Python code
- **Secret Detection**: Scans for hardcoded secrets and credentials

### File Validation

- **File Size**: Ensures files don't exceed size limits
- **Naming Conventions**: Validates file and directory naming
- **Content Checks**: Scans for TODO comments, debug code, etc.
- **Encoding**: Validates file encoding and line endings

## Integration

### With Global Workflows

The pre-commit hook integrates seamlessly with global workflows:

- **Python Development Workflow**: Uses same tool configurations
- **Git Workflow**: Complements branch management and PR preparation
- **CI/CD Pipelines**: Can be run as part of automated testing

### With MCP Servers

Hooks can leverage MCP servers for enhanced capabilities:

- **Context7 MCP**: For up-to-date security advisories
- **Fetch MCP**: For external security database queries
- **Project Health Auditor**: For code quality metrics

## Best Practices

### - Hook Configuration

1. **Start Simple**: Begin with basic checks and add complexity gradually
2. **Respect Project Configs**: Use existing project configuration files
3. **Exclude Appropriately**: Use exclude patterns for generated code, tests, etc.
4. **Performance**: Balance thoroughness with commit speed
5. **Team Consistency**: Share hook configurations across team members

### Development Workflow

1. **Install Early**: Set up hooks at the start of projects
2. **Test Thoroughly**: Ensure hooks don't break legitimate commits
3. **Document**: Include hook setup in project documentation
4. **Monitor**: Review hook output for false positives/negatives
5. **Update**: Keep hook configurations and tools updated

## Troubleshooting

### Common Issues

**Hook Not Running:**

- Check Git hook installation: `ls -la .git/hooks/pre-commit`
- Verify hook is executable: `chmod +x .git/hooks/pre-commit`
- Check Git configuration: `git config core.hooksPath`

**Tool Not Found:**

- Ensure tools are installed in the environment
- Check PATH includes tool locations
- Verify virtual environment activation

**Configuration Issues:**

- Check JSON syntax in configuration files
- Verify file paths and permissions
- Test with minimal configuration first

**Performance Issues:**

- Exclude large files and directories
- Use incremental checking when possible
- Consider running only essential checks

### Getting Help

- Check hook output for specific error messages
- Review configuration file syntax
- Test individual tools separately
- Use `--verbose` flag for detailed output

## Custom Hooks

### Creating Custom Hooks

To create your own global hook:

1. **Create Python script** in `~/.cline/global_hooks/`
2. **Follow naming convention**: Use descriptive names with `.py` extension
3. **Implement main function**: Include `if __name__ == "__main__":` block
4. **Add documentation**: Include docstring and usage examples
5. **Test thoroughly**: Ensure it works across different project types

### Hook Template

```python
#!/usr/bin/env python3
"""
Custom Hook Name

Description of what this hook does.
"""

import sys
import os
from pathlib import Path


def main():
    """Main entry point for the custom hook."""
    if len(sys.argv) < 2:
        print("Usage: python custom_hook.py <command>")
        print("Commands: install, run, status, uninstall")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "install":
        install_hook()
    elif command == "run":
        run_hook()
    elif command == "status":
        check_status()
    elif command == "uninstall":
        uninstall_hook()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def install_hook():
    """Install the hook."""
    print("Installing custom hook...")


def run_hook():
    """Run the hook."""
    print("Running custom hook...")


def check_status():
    """Check hook status."""
    print("Checking hook status...")


def uninstall_hook():
    """Uninstall the hook."""
    print("Uninstalling custom hook...")


if __name__ == "__main__":
    main()
```

## Contributing

### Adding New Hooks

1. **Create hook script** following the template
2. **Add documentation** to this README
3. **Test thoroughly** with different project types
4. **Submit PR** with clear description of the hook's purpose

### Improving Existing Hooks

1. **Report issues** with specific examples
2. **Suggest improvements** via PRs
3. **Add new features** while maintaining backward compatibility
4. **Update documentation** for any changes
