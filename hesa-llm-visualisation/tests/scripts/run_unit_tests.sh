#!/bin/bash

echo
echo "HESA LLM Visualization - Unit Tests"
echo "=================================="
echo

# Determine script directory and set PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if pytest is installed
if ! python -c "import pytest" &> /dev/null; then
    echo "[ERROR] pytest is not installed."
    echo "Please install pytest with: pip install pytest"
    exit 1
fi

echo "Running unit tests..."
echo

# Run pytest with appropriate options
python -m pytest -v "$PROJECT_ROOT/tests/unit/"

echo
echo "Unit tests completed!"
echo 