```instructions
---
applyTo: "airline-discount-ml/data/**/*.sql,data/**/*.sql,airline-discount-ml/src/data/**/*.py,src/data/**/*.py"
---

# Copilot Instructions for Database Schema and Data Layer

This file provides comprehensive guidance for working with the database schema and data layer to ensure schema integrity and prevent breaking changes.

## Purpose
- Protect database schema integrity across training exercises
- Ensure Copilot respects the fixed schema contract
- Prevent accidental schema changes that would break models and tests

## Scope
- `data/schema.sql`: SQLite schema definition (READ-ONLY)
- `data/sample_data.sql`: Sample data for initialization  
- `src/data/database.py`: Database connection and query helpers
- `src/data/preprocessor.py`: Data transformation utilities
- `src/data/load_synthetic_data.py`: Synthetic data generation

## Fixed Database Schema (DO NOT MODIFY)

The three-table schema (passengers, routes, discounts) is LOCKED for training consistency.

See full file content for:
- Complete schema definitions
- Hard constraints (what NOT to change)
- Safe operations guidelines  
- Coding standards for data layer
- Query best practices
- JSON travel_history contract
- Model dependencies on schema
- Testing requirements
- Common pitfalls and solutions

**Total:** 240+ lines of comprehensive data layer guidance
```
