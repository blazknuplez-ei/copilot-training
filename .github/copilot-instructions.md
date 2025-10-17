# GitHub Copilot Instructions - Training Repository

## Project Context
This is a **training repository** for teaching GitHub Copilot, MCP tools, and ML workflows. Contains two main components:
1. **Exercises** (`exercises/`) - Step-by-step training modules for learning Copilot, MCP servers, and testing
2. **Sample ML Project** (`airline-discount-ml/`) - Working airline discount prediction system with synthetic data

**Critical:** All data is synthetic. No real passenger information, no production secrets.

## Repository Structure

```
copilot-training/
├── .github/
│   ├── copilot-instructions.md          # This file (repo-wide rules)
│   └── instructions/
│       └── models.instructions.md       # Path-scoped rules for ML models
├── exercises/                           # Training curriculum
│   ├── 01-setup/                       # Copilot instructions setup
│   ├── 02-mcp-server-for-github/       # GitHub MCP server integration
│   ├── 03-custom-mcp-server/           # Build custom MCP with Synth
│   └── 04-unit-tests/                  # pytest with Copilot
└── airline-discount-ml/                 # Sample ML project
    ├── src/
    │   ├── models/      # Pure ML (no I/O) - see .github/instructions/models.instructions.md
    │   ├── data/        # Database layer (SQLite only)
    │   ├── agents/      # Business logic stubs for training
    │   ├── mcp/         # Custom MCP server implementation
    │   └── training/    # Train/evaluate scripts
    ├── tests/           # pytest suite (must pass before commits)
    ├── notebooks/       # Jupyter explorations
    └── data/            # SQLite DB + schema files
```

**Core Pattern:** `src.models` contains pure ML code (no I/O), `src.data.database` handles all database operations, `src.agents` implements discount calculation logic.

## Architecture & Data Flow

### 1. Database Layer (`src/data/`)
- **SQLite only** (no PostgreSQL) - `data/airline_discount.db`
- Schema: `passengers` (name, travel_history JSON), `routes` (origin, destination, distance), `discounts` (links passengers/routes with discount_value)
- Initialize with: `python -c "from src.data.database import init_database; init_database()"`
- All DB access goes through `Database` class or `get_connection()` helper

### 2. Models Layer (`src/models/`)
**Critical:** Models NEVER import from `src.data.database`. They are pure functions/classes taking DataFrames.

- `discount_predictor.py`: sklearn Pipeline with ColumnTransformer → LinearRegression
  - API: `fit(X, y)`, `predict(X)`, `save(path)`, `load(path)` (classmethod)
  - Required features: `['distance_km', 'history_trips', 'avg_spend', 'route_id', 'origin', 'destination']`
  - Numeric features: SimpleImputer(median) → StandardScaler
  - Categorical features: SimpleImputer(most_frequent) → OneHotEncoder
  - Always set `random_state=42` and module-level seeds: `random.seed(42); np.random.seed(42)`

- `passenger_profiler.py`: Pure function `build_features(df) -> DataFrame`
  - Converts miles to km: `distance * 1.60934`
  - Derives `avg_spend` from `total_spend / history_trips` if needed
  - Returns only required columns in correct order
  - No PII (no `passenger_id` in output)

### 3. Agents Layer (`src/agents/`)
Stub implementations for training exercises. Pattern: classes with business logic methods that could call models/database.

## Critical Architecture Decisions

### 1. Three-Layer Separation (ML Project)
- **Models Layer** (`src/models/`) - Pure functions/classes, no I/O, accepts DataFrames
  - NEVER import from `src.data.database` 
  - All models must be testable in isolation
  - Example: `DiscountPredictor.fit(X, y)` - no DB coupling
- **Data Layer** (`src/data/`) - All database operations, connection management
  - Only layer that touches SQLite
  - Provides `Database` class and `get_connection()` helper
  - Schema: passengers, routes, discounts tables
- **Agents Layer** (`src/agents/`) - Business logic, orchestrates models + data
  - Stub implementations for training exercises
  - Pattern: classes with methods that call models/database

**Why:** Enables unit testing, reproducibility, and clear boundaries. Models can be tested with synthetic DataFrames without database setup.

### 2. Cross-Platform Setup Scripts
- `setup.sh` (Mac/Linux) and `setup.bat` (Windows) provide identical workflows
- Both scripts: create venv → install deps → register Jupyter kernel → init DB
- **Critical:** Use `pip install -e ".[dev]"` (editable install) not just requirements.txt
- Enables `from src.models import X` without sys.path hacks (except in notebooks)

### 3. Path-Scoped Instructions
- `.github/copilot-instructions.md` - Repo-wide rules
- `.github/instructions/models.instructions.md` - Model-specific rules with `applyTo` glob
- Pattern: `applyTo: "airline-discount-ml/src/models/**/*.py,src/models/**/*.py"`
- See Exercise 01 for how to add additional scoped rules

## Development Workflows

### Setup (one-time)
```bash
cd airline-discount-ml
./setup.sh              # Mac/Linux
# OR: setup.bat         # Windows (PowerShell)
# OR: pip install -e ".[dev]"  # Manual
```

**What it does:**
1. Creates Python venv
2. Installs package in editable mode with dev dependencies
3. Registers "Python (airline-discount-ml)" Jupyter kernel
4. Initializes SQLite database with schema + sample data

**Verify:** `pytest tests/ -v` should pass all tests

### Tests
```bash
pytest tests/ -v              # Run all tests
pytest tests/test_models.py   # Specific file
make test-cov                 # With coverage
```

**Test Structure:**
- `tests/test_models.py`: Comprehensive pytest suite with fixtures
- Tests validate: fit/predict, baseline comparison (DummyRegressor), save/load roundtrip, index preservation
- 10 tests must pass before committing model changes

### Notebooks
- Start: `jupyter lab` or `make run-notebook` (from airline-discount-ml/)
- **Critical:** Select kernel "Python (airline-discount-ml)" not base Python
- Required first cell (notebooks aren't editable-installed):
  ```python
  import sys; from pathlib import Path
  sys.path.insert(0, str(Path().resolve().parent))
  from src.data.database import get_connection
  ```
- Sample notebook: `notebooks/exploratory_analysis.ipynb` shows DB connection pattern

### Database Operations
```bash
# From airline-discount-ml/ directory:
make db-init        # Reinitialize schema and sample data
make db-sample      # Load sample data only
python -c "from src.data.database import init_database; init_database()"  # Manual

# Check data: jupyter lab → notebooks/exploratory_analysis.ipynb → run cells
```

**DB Location:** `airline-discount-ml/data/airline_discount.db` (SQLite)
**Schema:** passengers (name, travel_history JSON), routes (origin, destination, distance), discounts (passenger/route/discount_value)

## Project-Specific Conventions

### 1. **Editable Install Pattern**
- Package installed with `pip install -e ".[dev]"` (not just requirements.txt)
- Enables `from src.models import DiscountPredictor` without sys.path hacks (except in notebooks)
- Dev dependencies in `setup.py` extras_require: pytest, jupyter, black, flake8
- **Why:** Allows imports to work immediately after editing code, no reinstall needed

### 2. **Deterministic ML**
- ALWAYS set `random_state=42` in sklearn estimators
- Set module-level seeds: `random.seed(42); np.random.seed(42)` at top of model files
- Required for reproducible training exercises
- **Example:** `LinearRegression()` doesn't need it, but `train_test_split(..., random_state=42)` does

### 3. **No PII in Models**
- Never use `passenger_id` as a feature (only for joins)
- Only schema-driven features: distance, history metrics, route info
- Travel history stored as JSON text, not referenced directly by models
- **Rationale:** Training data is synthetic, but models should be designed to avoid PII in production

### 4. **Validation-First Approach**
- All model methods validate inputs before processing
- Raise `ValueError` with clear messages for empty DataFrames, missing columns
- Raise `RuntimeError` if predict called before fit
- **Pattern:** `_validate_X(X)` and `_validate_y(y)` static methods in models

### 5. **Index Preservation**
- `predict()` must return Series with same index as input X
- Critical for joining predictions back to source data
- Tested explicitly in `test_predict_preserves_index`
- **Implementation:** `pd.Series(preds, index=X.index, name="discount_value")`

## Common Pitfalls & Solutions

### ❌ Import Error in Notebooks
**Problem:** `ModuleNotFoundError: No module named 'src'`
**Solution:** Add to first cell:
```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))
```
**Root Cause:** Notebooks run in `airline-discount-ml/notebooks/`, need parent in path

### ❌ Database Not Found
**Problem:** `sqlite3.OperationalError: no such table`
**Solution:** Run `make db-init` or `python -c "from src.data.database import init_database; init_database()"`
**Root Cause:** First-time setup didn't complete, or DB file deleted

### ❌ Model Predicting Before Fit
**Problem:** RuntimeError during predict
**Solution:** Check `self._fitted` flag, call `fit()` first
**Root Cause:** DiscountPredictor requires `fit(X, y)` before `predict(X)`

### ❌ Wrong Jupyter Kernel
**Problem:** Imports work in terminal but not notebook
**Solution:** Restart kernel, select "Python (airline-discount-ml)" kernel
**Root Cause:** Using base Python interpreter instead of venv kernel

## When to Use Which Tool

### MCP Tools (custom server in `src/mcp/`)
- `query_db`: Fetch data from SQLite for training
- `train_model`: Execute model training pipelines
- `predict`: Run predictions on new data
- `evaluate`: Calculate MAE, R² metrics

### Standard Commands
- **Format:** `black src tests` or `make format`
- **Lint:** `flake8 src tests` or `make lint`
- **Train:** `python src/training/train.py` or `make train`
- **Evaluate:** `python src/training/evaluate.py` or `make evaluate`

## File-Specific Guidance

When working in **`src/models/`**, follow `.github/instructions/models.instructions.md` for detailed API contracts, feature requirements, and testing checklist.

## Editing Guidelines

- **<2 files, <50 lines:** Proceed with changes
- **≥2 files OR ≥50 lines:** Propose plan first, wait for approval
- **API changes:** Always ask before modifying public interfaces (fit/predict signatures, function names)
- **Failing tests:** Fix the implementation code, NOT the tests. Tests define correct behavior.
- **Context priority:** selections > symbols > files > #codebase (use #codebase only if requested)

## Communication Style

- **Be concise:** Get to the point quickly. No verbose explanations unless asked.
- **Show, don't tell:** Use code examples instead of lengthy descriptions.
- **Skip preamble:** Don't repeat what the user already knows or said.

## Training Repository Philosophy

This repo demonstrates **professional ML project setup** for teaching purposes:
- Automated setup scripts (cross-platform)
- Makefile for command shortcuts
- Comprehensive tests with pytest
- Type hints and docstrings throughout
- Clear separation: data layer, model layer, agent layer
- No production secrets (synthetic data only)

If required columns are missing, ask for clarification instead of guessing.

**Goal:** Trainees learn GitHub Copilot best practices through well-structured, documented code.