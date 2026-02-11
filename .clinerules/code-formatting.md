## Brief overview
Comprehensive code formatting and linting ruleset for the FractalSemantics project. This ensures consistent code quality, readability, and maintainability across all Python files in the project.

## Whitespace management
  - **No whitespace on empty lines**: Empty lines must contain no characters, not even spaces or tabs
  - **Use fix_whitespace.py**: Run the automated whitespace cleanup script before committing changes
  - **Trailing whitespace removal**: Remove all trailing whitespace from the end of lines
  - **Consistent indentation**: Use 4 spaces for Python indentation, never mix tabs and spaces
  - **Empty line formatting**: Ensure empty lines are truly empty for better diff readability

## Import organization
  - **Standard library first**: Group imports in this order: standard library, third-party, local modules
  - **Alphabetical sorting**: Sort imports alphabetically within each group
  - **Blank line separation**: Separate import groups with single blank lines
  - **No wildcard imports**: Avoid `from module import *` - import specific functions/classes
  - **Relative imports**: Use relative imports for local modules (e.g., `from .module import function`)

## Code style and formatting
  - **Line length**: Maximum 88 characters per line (Black formatter default)
  - **String quotes**: Use double quotes for strings, single quotes for internal strings
  - **Function spacing**: Two blank lines before class definitions, one before function definitions
  - **Method spacing**: One blank line between methods in classes
  - **Operator spacing**: Add spaces around operators (`x = y + z`, not `x=y+z`)

## Documentation standards
  - **Docstrings required**: All public functions, classes, and modules must have docstrings
  - **Google style**: Use Google-style docstrings with Args, Returns, Raises sections
  - **Type hints**: Include type hints for all function parameters and return values
  - **Inline comments**: Use inline comments for complex logic explanations
  - **Module documentation**: Each module should have a module-level docstring

## Error handling
  - **Try-catch blocks**: Use specific exception types, avoid bare `except:` clauses
  - **User-friendly messages**: Provide clear, actionable error messages for users
  - **Logging**: Use structured logging for debugging and monitoring
  - **Graceful degradation**: Handle errors gracefully without crashing the application
  - **Input validation**: Validate all user inputs and external data

## Testing requirements
  - **Test coverage**: Maintain minimum 80% test coverage for all modules
  - **Test naming**: Use descriptive test names that explain what is being tested
  - **Test isolation**: Each test should be independent and not depend on other tests
  - **Mock external dependencies**: Use mocking for external services and APIs
  - **Integration tests**: Include integration tests for critical workflows

## Performance considerations
  - **Resource monitoring**: Monitor memory usage and CPU consumption
  - **Efficient algorithms**: Choose appropriate data structures and algorithms
  - **Lazy loading**: Implement lazy loading for large datasets and expensive operations
  - **Caching strategies**: Use caching for expensive computations and database queries
  - **Memory management**: Clean up resources and avoid memory leaks

## Git commit standards
  - **Clear commit messages**: Use descriptive commit messages with context
  - **Atomic commits**: Each commit should represent a single logical change
  - **Pre-commit hooks**: Use pre-commit hooks to enforce formatting and linting
  - **Branch naming**: Use descriptive branch names (e.g., `feature/gui-improvements`)
  - **Pull request reviews**: Require code reviews for all changes

## Security best practices
  - **Input sanitization**: Sanitize all user inputs to prevent injection attacks
  - **Secrets management**: Never commit API keys, passwords, or sensitive data
  - **Dependency updates**: Keep dependencies updated to patch security vulnerabilities
  - **Error information**: Don't expose sensitive information in error messages
  - **Access controls**: Implement proper authentication and authorization

## Development workflow
  - **Research before code**: Always check planned referenced code before writing new code
  - **Test before commit**: Always run tests before committing changes
  - **Code review**: Submit changes for review before merging to main branch
  - **Continuous integration**: Ensure CI/CD pipeline passes all checks
  - **Documentation updates**: Update documentation when adding new features
  - **Backward compatibility**: Maintain backward compatibility when possible

## File organization
  - **Logical grouping**: Group related functionality in appropriate modules
  - **Clear naming**: Use descriptive file and directory names
  - **Separation of concerns**: Keep different responsibilities in separate modules
  - **Configuration management**: Use configuration files for environment-specific settings
  - **Resource management**: Organize static files, templates, and assets logically