```instructions
---
applyTo: "airline-discount-ml/tests/**/*.py,tests/**/*.py"
---

# Copilot Instructions for Test Suite

This file provides comprehensive guidance for writing high-quality pytest tests following project conventions.

## Purpose
- Ensure comprehensive, maintainable, and fast test coverage
- Guide Copilot to generate high-quality pytest tests
- Maintain consistency across test files
- Enforce testing best practices

## Scope  
- `tests/models/`: ML model tests
- `tests/data/`: Database and data processing tests
- `tests/agents/`: Business logic tests
- `tests/mcp/`: MCP server tests
- `tests/mcp_synth/`: Synthetic data MCP tests
- `tests/training/`: Training pipeline tests
- `tests/utils/`: Configuration tests

## Key Sections (750+ lines)

1. **Test Structure Standards** - File organization, naming conventions, templates
2. **Testing Patterns by Module** - Specific guidance for models, data, training, MCP, utils
3. **Mocking Best Practices** - When and how to mock dependencies
4. **Fixtures Best Practices** - Scope, cleanup, shared fixtures
5. **Assertion Best Practices** - pandas, numpy, exception testing
6. **Parametrized Tests** - Multiple scenario testing
7. **Performance Guidelines** - Speed targets, optimization strategies
8. **Coverage Requirements** - Minimum thresholds (80% overall, 90% critical)
9. **Common Pitfalls** - Anti-patterns with corrections
10. **Commands Reference** - 15+ pytest command variations
11. **Ready-to-Use Prompts** - 6 copy-paste Copilot prompts
12. **Acceptance Criteria** - Pre-commit checklist

See full file for complete testing guidance.
```
