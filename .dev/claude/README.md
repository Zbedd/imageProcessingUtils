# Claude - Agentic Verification Tests

This folder contains tests created by Claude to verify that structural changes and refactoring were successful. These are **not** production tests for the actual utility functions.

## Purpose

- Verify imports work after structural changes
- Test that modules can be loaded correctly
- Validate package structure integrity
- Ensure configuration files are properly set up

## Important Notes

- **This folder is gitignored** - these tests are not part of the production codebase
- These tests are for development/refactoring verification only
- Production tests should go in the main `tests/` directory
- Run these tests after making structural changes to verify everything still works

## Usage

```bash
cd claude
python test_imports.py
```

or 

```bash
pytest claude/ -v
```
