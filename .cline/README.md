# Cline Development System

This directory contains the Cline development system for the FractalSemantics project. Cline provides powerful slash commands for development workflows, code quality, and project management.

## Available Slash Commands

### Workflows

#### `/python-dev-workflow`
Comprehensive Python development workflow automation.

**Usage:**
- `/python-dev-workflow` - Run full development workflow
- `/python-dev-workflow setup` - Setup development environment
- `/python-dev-workflow check` - Run quality checks only
- `/python-dev-workflow docs` - Generate documentation
- `/python-dev-workflow deps` - Check dependencies

**Features:**
- Environment setup with virtual environments
- Code quality checks (Ruff, Black, MyPy)
- Security scanning (Safety, Bandit)
- Testing with pytest and coverage
- Documentation generation with Sphinx
- Dependency management

**Configuration:** `.cline-workflow-config.json`

#### `/git-workflow`
Comprehensive Git workflow automation.

**Usage:**
- `/git-workflow` - Show available commands
- `/git-workflow setup-hooks` - Setup pre-commit hooks
- `/git-workflow feature <name>` - Create feature branch
- `/git-workflow release <version>` - Create release branch
- `/git-workflow pr` - Prepare pull request
- `/git-workflow release-tag <version>` - Create release tag
- `/git-workflow status` - Check repository status
- `/git-workflow cleanup` - Clean up merged branches
- `/git-workflow changelog [tag]` - Generate changelog

**Features:**
- Pre-commit hooks setup
- Branch management (feature, release, hotfix)
- Pull request preparation
- Release management and tagging
- Status checking and cleanup
- Changelog generation

**Configuration:** `.cline-git-config.json`

### Hooks

#### `/pre-commit-hook`
Comprehensive pre-commit hook for code quality.

**Usage:**
- `/pre-commit-hook` - Show available commands
- `/pre-commit-hook setup` - Setup pre-commit hook for current repository
- `/pre-commit-hook check` - Run quality checks on staged files
- `/pre-commit-hook config` - Show configuration

**Features:**
- Linting checks (Ruff, Flake8, ESLint)
- Code formatting checks (Black, Prettier)
- Type checking (MyPy, TypeScript)
- Security scanning (Bandit, Safety)
- Test execution
- Commit message validation

**Configuration:** `.cline-pre-commit.json`

### Skills

#### `/code-reviewer`
AI-powered code reviewer for multiple programming languages.

**Usage:**
- `/code-reviewer <file_path>` - Review single file
- `/code-reviewer .` - Review entire project
- `/code-reviewer <file> --security` - Security-focused review
- `/code-reviewer <file> --performance` - Performance-focused review
- `/code-reviewer <file> --best-practices` - Best practices review

**Features:**
- Security vulnerability detection
- Performance issue identification
- Code style and formatting checks
- Best practices validation
- Type safety analysis
- Multi-language support (Python, JavaScript, Rust, Go, Java, C/C++)

**Configuration:** `.cline-code-reviewer.json`

#### `/project-analyzer`
Comprehensive project analysis tool.

**Usage:**
- `/project-analyzer` - Analyze current project
- `/project-analyzer <path>` - Analyze specific project
- `/project-analyzer --metrics` - Show detailed metrics
- `/project-analyzer --dependencies` - Show dependency analysis
- `/project-analyzer --structure` - Show project structure
- `/project-analyzer --languages` - Show language analysis

**Features:**
- Project structure analysis
- File and line metrics
- Dependency analysis
- Language usage patterns
- Code quality indicators
- Development recommendations

**Configuration:** `.cline-project-analyzer.json`

## Configuration Files

Each command has its own configuration file for customization:

- `.cline-workflow-config.json` - Python development workflow settings
- `.cline-git-config.json` - Git workflow settings
- `.cline-pre-commit.json` - Pre-commit hook settings
- `.cline-code-reviewer.json` - Code reviewer settings
- `.cline-project-analyzer.json` - Project analyzer settings

## Setup

1. **Install Dependencies:**
   ```bash
   # For Python development
   pip install ruff black mypy pytest safety bandit
   
   # For Git workflows
   pip install pre-commit
   
   # For documentation
   pip install sphinx
   ```

2. **Setup Git Hooks:**
   ```bash
   /git-workflow setup-hooks
   ```

3. **Configure Pre-commit:**
   ```bash
   /pre-commit-hook setup
   ```

4. **Run Initial Analysis:**
   ```bash
   /project-analyzer
   ```

## Best Practices

1. **Use Workflows Regularly:**
   - Run `/python-dev-workflow check` before commits
   - Use `/git-workflow pr` for pull request preparation
   - Run `/python-dev-workflow` for comprehensive checks

2. **Code Review:**
   - Use `/code-reviewer` for critical files
   - Focus on security and performance reviews
   - Address all high-priority issues

3. **Project Management:**
   - Use `/project-analyzer` to understand project structure
   - Follow generated recommendations
   - Monitor code quality metrics

4. **Git Workflow:**
   - Use feature branches for development
   - Prepare pull requests with `/git-workflow pr`
   - Clean up branches regularly with `/git-workflow cleanup`

## Troubleshooting

### Common Issues

1. **Command Not Found:**
   - Ensure Cline is properly installed
   - Check that the `.cline` directory exists
   - Verify Python scripts are executable

2. **Missing Dependencies:**
   - Install required tools (ruff, black, mypy, etc.)
   - Check Python environment
   - Verify PATH configuration

3. **Configuration Errors:**
   - Validate JSON syntax in config files
   - Check file permissions
   - Use default configurations as reference

### Getting Help

- Use `--help` flag with any command
- Check individual command documentation
- Review configuration file examples
- Consult Cline documentation

## Integration

### IDE Integration

These commands work seamlessly with:
- VS Code
- PyCharm
- Vim/Neovim
- Emacs
- Any terminal-based editor

### CI/CD Integration

Commands can be integrated into CI/CD pipelines:
```yaml
# GitHub Actions example
- name: Run Python Development Workflow
  run: /python-dev-workflow check

- name: Code Review
  run: /code-reviewer . --security
```

### Automation

Set up automated workflows:
- Pre-commit hooks for quality checks
- Scheduled project analysis
- Automated dependency updates
- Regular security scans

## Contributing

To contribute to the Cline system:

1. **Add New Commands:**
   - Create Python scripts in appropriate directories
   - Follow existing command patterns
   - Add comprehensive help documentation

2. **Improve Existing Commands:**
   - Enhance error handling
   - Add new features
   - Improve performance

3. **Update Documentation:**
   - Keep README current
   - Add usage examples
   - Document configuration options

## License

This Cline system is part of the FractalSemantics project and follows the same licensing terms.