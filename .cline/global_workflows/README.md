# Global Workflows for Cline

This directory contains global workflows that can be used across all Cline workspaces to automate common development tasks.

## Available Workflows

### Python Development Workflow (`python_dev_workflow.py`)

A comprehensive workflow for Python projects that includes:

- **Environment Setup**: Creates and configures virtual environments
- **Code Quality Checks**: Runs Ruff, Black, and MyPy
- **Security Scanning**: Uses Safety and Bandit for security analysis
- **Testing**: Executes pytest with coverage reporting
- **Documentation**: Builds Sphinx documentation
- **Dependency Management**: Checks for outdated packages

**Usage:**
```bash
python ~/.cline/global_workflows/python_dev_workflow.py
```

**Features:**
- Automatic virtual environment detection and creation
- Configurable tool versions and settings
- Comprehensive error reporting
- Integration with existing project configurations

### Git Workflow (`git_workflow.py`)

Comprehensive Git workflow automation including:

- **Pre-commit Hooks Setup**: Configures pre-commit hooks for code quality
- **Branch Management**: Creates feature and release branches
- **Pull Request Preparation**: Updates branches and runs checks
- **Release Management**: Creates tags and release branches
- **Status Checking**: Provides Git status and recommendations
- **Branch Cleanup**: Removes merged branches
- **Changelog Generation**: Creates changelogs from Git commits

**Usage:**
```bash
# Setup pre-commit hooks
python ~/.cline/global_workflows/git_workflow.py setup-hooks

# Create feature branch
python ~/.cline/global_workflows/git_workflow.py feature my-new-feature

# Prepare pull request
python ~/.cline/global_workflows/git_workflow.py pr

# Create release
python ~/.cline/global_workflows/git_workflow.py release-tag v1.0.0
```

**Commands:**
- `setup-hooks`: Install pre-commit hooks
- `feature <name>`: Create feature branch
- `release <version>`: Create release branch
- `pr`: Prepare pull request
- `release-tag <version>`: Create release tag
- `status`: Check Git status
- `cleanup`: Clean up merged branches
- `changelog [tag]`: Generate changelog

## Configuration

### Pre-commit Configuration

The Git workflow creates a `.pre-commit-config.yaml` file with the following tools:

- **pre-commit-hooks**: Basic file checks (trailing whitespace, YAML validation, etc.)
- **Black**: Python code formatting
- **Flake8**: Python linting
- **isort**: Import sorting

### Python Development Configuration

The Python workflow uses the following default versions:
- Black: Latest
- Ruff: Latest
- MyPy: Latest
- pytest: Latest
- Safety: Latest
- Bandit: Latest

## Integration

These workflows are designed to work seamlessly with:

- **Existing project configurations**: Respects existing `.pre-commit-config.yaml`, `pyproject.toml`, etc.
- **CI/CD pipelines**: Can be integrated into GitHub Actions, GitLab CI, etc.
- **IDE integrations**: Works with VS Code, PyCharm, and other editors
- **Package managers**: Supports pip, poetry, conda environments

## Customization

### Python Workflow Customization

Create a `cline-workflow-config.json` file in your project root:

```json
{
  "python_dev_workflow": {
    "dev_requirements": [
      "custom-package>=1.0.0"
    ],
    "tool_versions": {
      "black": "22.3.0",
      "ruff": "0.1.0"
    },
    "skip_steps": ["documentation", "dependency_check"]
  }
}
```

### Git Workflow Customization

Create a `.cline-git-config.json` file:

```json
{
  "git_workflow": {
    "base_branch": "main",
    "release_branch_prefix": "release/",
    "feature_branch_prefix": "feature/",
    "pre_commit_hooks": [
      "custom-hook"
    ]
  }
}
```

## Best Practices

1. **Run workflows regularly**: Integrate into your development workflow
2. **Customize for your project**: Adjust configurations to match project needs
3. **Use in CI/CD**: Run workflows in your continuous integration pipeline
4. **Keep tools updated**: Regularly update the tools used by workflows
5. **Review output**: Always review workflow output for issues

## Troubleshooting

### Common Issues

**Virtual Environment Not Found:**
- Ensure Python is installed and in PATH
- Check that the virtual environment directory exists
- Verify permissions on the project directory

**Tool Installation Failures:**
- Check internet connectivity
- Verify package repository access
- Try installing tools manually first

**Git Workflow Errors:**
- Ensure Git is installed and configured
- Check repository status and permissions
- Verify remote repository access

### Getting Help

- Check the workflow output for specific error messages
- Review tool documentation for configuration options
- Consult project-specific configuration files
- Use the `--help` flag with workflow commands for usage information