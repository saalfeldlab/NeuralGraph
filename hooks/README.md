# Git Hooks

This directory contains git hooks for the NeuralGraph repository.

## Installation

To install the hooks, run:

```bash
./hooks/install.sh
```

## Pre-commit Hook

The pre-commit hook performs the following checks:

### 1. File Size Check

Prevents accidentally committing large files:
- **Regular files**: Maximum 500KB (configurable)
- **Notebook files (*.ipynb)**: Maximum 2MB (configurable)

To configure the size limits, edit the `MAX_FILE_SIZE` and `MAX_NOTEBOOK_SIZE` variables in `.git/hooks/pre-commit` after installation.

### 2. Ruff Linting

Runs `ruff check` on all Python files (excluding notebooks) using the conda environment:
- First checks for `neural-graph-linux` environment
- Falls back to `neural-graph-mac` if not found
- Exits with error if neither environment is available

If ruff check fails, the hook will provide the exact command to fix the issues:

```bash
conda run -n <env-name> ruff check . --exclude '*.ipynb' --fix
```

Or you can activate the environment and run ruff directly:

```bash
conda activate <env-name>
ruff check . --exclude '*.ipynb' --fix
```

## Skipping the Hook (Not Recommended)

If you need to skip the pre-commit hook for a specific commit, use:

```bash
git commit --no-verify
```

**Note:** This should only be used in exceptional circumstances.
