# Global Skills for Cline

This directory contains global skills that can be used across all Cline workspaces to enhance development capabilities.

## Available Skills

### Code Reviewer (`code_reviewer.py`)

An AI-powered code reviewer that provides comprehensive feedback on code quality, security, performance, and best practices.

**Features:**

- **Security Analysis**: Detects hardcoded secrets, SQL injection, XSS vulnerabilities
- **Performance Review**: Identifies inefficient loops, string concatenation, N+1 queries
- **Code Style**: Checks line length, trailing whitespace, naming conventions
- **Best Practices**: Reviews mutable defaults, error handling, docstrings
- **Multi-language Support**: Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, HTML/CSS

**Usage:**

```bash
# Review single file
python ~/.cline/global_skills/code_reviewer.py src/main.py

# Review entire project
python ~/.cline/global_skills/code_reviewer.py .

# Review with custom project root
python ~/.cline/global_skills/code_reviewer.py . /path/to/project
```

**Output:**

- Detailed issue reports with severity levels
- Specific line numbers and suggestions
- Categorized by error, warning, and info
- Summary statistics

### Project Analyzer (`project_analyzer.py`)

Comprehensive project analysis tool that provides insights into project structure, technology stack, and code quality.

**Features:**

- **Technology Stack Detection**: Automatically detects languages, frameworks, and tools
- **Code Quality Metrics**: Analyzes complexity, naming conventions, code smells
- **Security Analysis**: Scans for hardcoded secrets and vulnerabilities
- **Performance Analysis**: Identifies bottlenecks and inefficient patterns
- **Dependency Analysis**: Reviews Python and JavaScript dependencies
- **Documentation Assessment**: Evaluates README quality and docstring coverage

**Usage:**

```bash
# Analyze current project
python ~/.cline/global_skills/project_analyzer.py

# Analyze specific project
python ~/.cline/global_skills/project_analyzer.py /path/to/project
```

**Output:**

- Technology stack confidence scores
- Quality score (0-100)
- Issue categorization and recommendations
- Detailed JSON report saved to `project_analysis_report.json`

## Integration

### With Workflows

These skills integrate seamlessly with global workflows:

- **Python Development Workflow**: Uses code reviewer for quality checks
- **Git Workflow**: Integrates project analyzer for pre-commit checks
- **CI/CD Pipelines**: Can be run as part of automated testing

### With MCP Servers

Skills can leverage MCP servers for enhanced capabilities:

- **Context7 MCP**: For up-to-date documentation and best practices
- **Fetch MCP**: For external research and information gathering
- **Project Health Auditor**: For code quality metrics

## Configuration

### Code Reviewer Configuration

Create a `.code-reviewer-config.json` file in your project root:

```json
{
  "code_reviewer": {
    "enabled_checks": [
      "security",
      "performance", 
      "style",
      "best_practices"
    ],
    "severity_threshold": "warning",
    "exclude_patterns": [
      "test_*",
      "*_test.py",
      "generated/*"
    ],
    "custom_rules": [
      {
        "pattern": "print\\(",
        "message": "Avoid print statements in production code",
        "severity": "info"
      }
    ]
  }
}
```

### Project Analyzer Configuration

Create a `.project-analyzer-config.json` file:

```json
{
  "project_analyzer": {
    "exclude_patterns": [
      "node_modules/**",
      ".git/**",
      "dist/**",
      "build/**"
    ],
    "tech_stack_weights": {
      "Python": 10,
      "JavaScript": 8,
      "Docker": 5
    },
    "quality_thresholds": {
      "docstring_coverage": 80,
      "comment_density": 10,
      "max_file_size": 1000
    }
  }
}
```

## Best Practices

### Using Code Reviewer

1. **Run before commits**: Use as part of pre-commit checks
2. **Review critical code**: Focus on security-sensitive and performance-critical code
3. **Customize rules**: Adapt to your project's coding standards
4. **Integrate with CI**: Run in continuous integration pipeline
5. **Team adoption**: Share configuration across team members

### Using Project Analyzer

1. **Regular analysis**: Run weekly or monthly for ongoing projects
2. **Onboarding tool**: Use for new team members to understand project structure
3. **Refactoring planning**: Identify areas that need refactoring
4. **Architecture decisions**: Use insights for technology choices
5. **Documentation**: Generate reports for stakeholders

## Custom Skills

### Creating Custom Skills

To create your own global skill:

1. **Create Python script** in `~/.cline/global_skills/`
2. **Follow naming convention**: Use descriptive names with `.py` extension
3. **Implement main function**: Include `if __name__ == "__main__":` block
4. **Add documentation**: Include docstring and usage examples
5. **Test thoroughly**: Ensure it works across different project types

### Skill Template

```python
#!/usr/bin/env python3
"""
Custom Skill Name

Description of what this skill does.
"""

import sys
from pathlib import Path
from typing import dict, list, any


def main():
    """Main entry point for the custom skill."""
    if len(sys.argv) < 2:
        print("Usage: python custom_skill.py <argument>")
        sys.exit(1)
    
    argument = sys.argv[1]
    
    # Your skill logic here
    print(f"Processing: {argument}")
    
    # Return results
    results = {
        "status": "success",
        "data": {"processed": argument}
    }
    
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

**Permission Errors:**

- Ensure scripts are executable: `chmod +x script.py`
- Check file permissions in `~/.cline/global_skills/`

**Import Errors:**

- Verify Python path includes the skills directory
- Check for missing dependencies

**Configuration Issues:**

- Ensure JSON configuration files are valid
- Check file paths and permissions

### Getting Help

- Check skill output for specific error messages
- Review configuration file syntax
- Test with simple examples first
- Use `--help` flag if implemented

## Contributing

### Adding New Skills

1. **Create skill script** following the template
2. **Add documentation** to this README
3. **Test thoroughly** with different project types
4. **Submit PR** with clear description of the skill's purpose

### Improving Existing Skills

1. **Report issues** with specific examples
2. **Suggest improvements** via PRs
3. **Add new features** while maintaining backward compatibility
4. **Update documentation** for any changes
