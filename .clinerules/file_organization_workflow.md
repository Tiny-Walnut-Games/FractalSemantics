# File Organization and Large File Management Workflow

## Overview

This workflow addresses the issue of large files (>1000 lines) that become difficult to search and maintain. Based on analysis of the current project structure, we have several files exceeding 1000 lines that need better organization.

## Current Large File Analysis

**Files exceeding 1000 lines:**
- `fractalsemantics/exp20_vector_field_derivation.py` (1,367 lines)
- `fractalsemantics/exp03_coordinate_entropy.py` (1,075 lines)
- `fractalsemantics/exp19_orbital_equivalence.py` (1,054 lines)
- `fractalsemantics/exp10_multidimensional_query.py` (1,027 lines)
- `fractalsemantics/exp06_entanglement_detection.py` (1,019 lines)

## Workflow Implementation

### 1. File Decomposition Strategy

**Rule: Single Responsibility Principle for Files**
- Each file should have a maximum of 500 lines
- Files >300 lines should be reviewed for decomposition opportunities
- Related functionality should be grouped into logical modules

**Decomposition Pattern:**
```
expXX_main_functionality.py (core logic, <500 lines)
├── expXX_data_models.py (data structures, enums)
├── expXX_algorithms.py (algorithm implementations)
├── expXX_utils.py (helper functions)
└── expXX_validation.py (validation and testing)
```

### 2. Modular Architecture Guidelines

**Rule: Clear Module Boundaries**
- Each module should have a single, well-defined purpose
- Use clear import statements to show dependencies
- Maintain backward compatibility during refactoring

**Example Structure for exp20:**
```
fractalsemantics/exp20_vector_field_derivation/
├── __init__.py
├── main.py (core vector field logic)
├── coordinate_systems.py (coordinate transformations)
├── field_calculations.py (field computation algorithms)
├── validation.py (validation and error handling)
└── utils.py (utility functions)
```

### 3. Search Optimization

**Rule: Enhanced Searchability**
- Use consistent naming conventions
- Add comprehensive docstrings to all functions
- Implement structured logging for debugging
- Create index files for large modules

**Search Enhancement Pattern:**
```python
# Add to each module
MODULE_INDEX = {
    "classes": ["VectorField", "CoordinateSystem"],
    "functions": ["calculate_field", "transform_coordinates"],
    "constants": ["FIELD_THRESHOLD", "COORDINATE_PRECISION"],
    "description": "Vector field derivation and coordinate transformations"
}
```

### 4. Documentation Standards

**Rule: Inline Documentation**
- Every function >10 lines must have docstring
- Complex algorithms need inline comments
- Module-level documentation required for files >100 lines
- Use type hints for better IDE support

### 5. Testing Strategy

**Rule: Modular Testing**
- Each decomposed module gets its own test file
- Integration tests for module interactions
- Performance tests for critical algorithms
- Maintain test coverage >80%

## Implementation Steps

### Phase 1: Analysis (Current)
- [x] Identify large files
- [x] Analyze file structure and dependencies
- [ ] Create decomposition plan for each large file

### Phase 2: Refactoring
- [ ] Implement modular structure for exp20
- [ ] Create clear module boundaries
- [ ] Maintain backward compatibility
- [ ] Update import statements

### Phase 3: Optimization
- [ ] Add comprehensive documentation
- [ ] Implement search optimization patterns
- [ ] Create module index files
- [ ] Update testing strategy

### Phase 4: Validation
- [ ] Run existing tests to ensure compatibility
- [ ] Performance benchmarking
- [ ] Code review and approval
- [ ] Documentation updates

## Tools and Skills Integration

### MCP Server Usage
- Use Context7 MCP for library documentation when refactoring
- Use Fetch MCP for external documentation research
- Implement project-health-auditor for code quality checks

### Skills Integration
- Apply technical documentation skills for improved docstrings
- Use code organization skills for better module structure
- Implement search optimization skills for enhanced findability

## Success Metrics

- File size reduction: Target <500 lines per file
- Search efficiency: 50% faster code location
- Maintainability: Clear module boundaries
- Test coverage: Maintain >80% coverage
- Performance: No degradation in execution time

## Maintenance

- Monthly review of file sizes
- Quarterly refactoring of growing modules
- Continuous documentation updates
- Regular dependency analysis