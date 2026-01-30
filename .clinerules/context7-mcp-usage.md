## Brief overview
Guidelines for efficiently using the Context7 MCP server to access up-to-date library documentation and code examples, particularly for Python development. This rule ensures we leverage the most current information without overloading the workflow.

## Context7 MCP server usage
  - Always use Context7 MCP when working with external libraries, frameworks, or APIs
  - Prioritize using `query-docs` tool with specific library IDs for targeted documentation
  - Use `resolve-library-id` tool when unsure of exact library name or ID
  - Default to Python libraries when language context is ambiguous but Python is likely

## Efficient documentation access
  - Check Context7 MCP proactively when starting new features that involve external dependencies
  - Use library-specific queries rather than generic searches for better results
  - Prefer exact library names (e.g., "requests", "numpy", "pandas") over generic terms
  - Include version information in queries when specific versions are required

## Workflow integration
  - Use Context7 MCP before implementing unfamiliar library features
  - Check for latest best practices and patterns when working with established libraries
  - Verify API changes or deprecations when updating dependencies
  - Use for troubleshooting and understanding error messages related to external libraries

## Query optimization
  - Be specific about the task or feature being implemented
  - Include relevant context about the project type (web app, data analysis, etc.)
  - Use clear, descriptive queries that match the intended use case
  - Examples:
    - "How to implement JWT authentication with FastAPI"
    - "Best practices for pandas DataFrame operations in data analysis"
    - "Setting up async database connections with SQLAlchemy"

## Balance and efficiency
  - Use Context7 MCP when it adds clear value to the development process
  - Avoid overusing for trivial or well-known operations
  - Prioritize queries that prevent errors, improve performance, or follow current best practices
  - Consider the trade-off between research time and implementation time