#!/bin/bash
# Installation script for git hooks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DIR="$(git rev-parse --git-dir 2>/dev/null)"

if [ -z "$GIT_DIR" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

echo "Installing git hooks..."

# Install pre-commit hook
if [ -f "$SCRIPT_DIR/pre-commit" ]; then
    cp "$SCRIPT_DIR/pre-commit" "$GIT_DIR/hooks/pre-commit"
    chmod +x "$GIT_DIR/hooks/pre-commit"
    echo "✓ Installed pre-commit hook"
else
    echo "✗ pre-commit hook not found in $SCRIPT_DIR"
    exit 1
fi

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will:"
echo "  1. Check that files don't exceed size limits (500KB for regular files, 2MB for notebooks)"
echo "  2. Run ruff check using the neural-graph conda environment"
echo ""
echo "To configure file size limits, edit the MAX_FILE_SIZE and MAX_NOTEBOOK_SIZE variables in:"
echo "  $GIT_DIR/hooks/pre-commit"
