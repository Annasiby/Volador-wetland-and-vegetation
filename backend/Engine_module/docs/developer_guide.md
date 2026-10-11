# GIS Wetland Degradation & Depletion Engine — Developer Guide

This document is for software developers and systems engineers maintaining, extending, or integrating the GIS Wetland Degradation & Depletion Engine.

---

## 1. Codebase Architecture Principles

### 1. Frozen Scientific Core
Scientific algorithms (U-Net model weights, band calculations, CVA formulas, NDWI calculations, fusion weights) are **FROZEN** at Phase 12.
- **Do NOT modify** scientific logic inside modules (e.g. `src/gis_engine/change/`, `src/gis_engine/wetland/`, `src/gis_engine/metrics/`) without explicit scientific review.
- **Phase 13 changes** focus strictly on production wrapper logic, configurations, exception handling, logging, documentation, and reporting.

### 2. Single Source of Truth (SSOT)
All modules must import categorical class IDs, transition IDs, NoData constants, and channel order lists directly from `src/gis_engine/core/constants.py`.
- **Never duplicate** class definitions or transition mappings in new scripts.
- **Add assertions** in `constants.py` when introducing new categorical codes to guarantee they do not collide with `NODATA_CATEGORICAL` (value 255).

### 3. Dynamic Resolution of Spatial Grids
- **Do NOT use hardcoded date strings** (like `2023-08-14_2023-08-28`) as path fallbacks.
- Downstream modules must receive aligned directories and file paths dynamically resolved by the pipeline.
- The pipeline writes aligned intermediate bands (e.g. `green.tif`, `nir.tif`) to `outputs/runs/<run_id>/alignment/` and supplies these paths to downstream modules.

---

## 2. Developer Workflow & Coding Standards

### Exception Handling Guidelines
All engine-specific exceptions must inherit from the base `GISProcessingError` inside `src/gis_engine/core/exceptions.py`.
- Use specific sub-classes (`InputValidationError`, `ConfigurationError`, etc.) to fail fast and bubble up descriptive error messages to the CLI.
- **Never write silent catches** (e.g. `except Exception: pass`). Use logged warnings instead, identifying when optional metadata (like Shapefile projection info) is missing and defaulting gracefully with clear log visibility.

### Structured Logging Guidelines
Prepend all log statements with step-specific prefixes inside the pipeline.
- Log level should default to `INFO` in production, writing to `logs/analysis.log` inside the run output directory.
- The logging handler `fh` must be closed and removed inside a `finally` block in `run_analysis` to prevent file descriptor leaks and lock conditions on Windows systems.

---

## 3. Running and Adding Tests

The test suite uses `pytest`. The code coverage target is 68%+.

### Running Tests
Execute the entire test suite from the workspace root:
```bash
python -m pytest tests/ -v
```

Run tests with coverage reporting:
```bash
python -m pytest tests/ --cov=src -v
```

### New Phase 13 Test Modules
- `tests/test_phase13_ssot.py`: Confirms integrity and distinctness of the centralized constants.
- `tests/test_phase13_paths.py`: Verifies that no executable code paths fallback to hardcoded dates.
- `tests/test_phase13_exceptions.py`: Validates the complete exception hierarchy.
- `tests/test_phase13_pipeline.py`: Exercises successful pipeline orchestration and validation.
- `tests/test_phase13_api.py`: Validates public API service behavior.
- `tests/test_phase13_cli.py`: Exercises CLI argument parsing and exit codes.

### Coding Standards for New Tests
- **Use Mock Fixtures**: Always use mock fixtures for ONNXModelWrapper (`predict` and `load_model`) to keep unit tests fast and environment-independent. Do not require GPU/CUDA or a real model file to run standard unit tests.
- **Use Temporary Directories**: Always write outputs to `tempfile.TemporaryDirectory()` to prevent test debris.
- **Assert Directory Structure**: When testing the pipeline, verify the creation of all required files (manifest, metadata, reports) inside the output directory structure.
