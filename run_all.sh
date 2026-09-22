#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "OptionMC verification"
echo "Project: $PROJECT_DIR"

if [[ -x ".venv/Scripts/python.exe" ]]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
    VENV_PYTHON=".venv/bin/python"
else
    if command -v python3 >/dev/null 2>&1; then
        SYSTEM_PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then
        SYSTEM_PYTHON="python"
    else
        echo "Error: Python 3 was not found on PATH." >&2
        exit 1
    fi

    echo
    echo "Creating virtual environment..."
    "$SYSTEM_PYTHON" -m venv .venv

    if [[ -x ".venv/Scripts/python.exe" ]]; then
        VENV_PYTHON=".venv/Scripts/python.exe"
    else
        VENV_PYTHON=".venv/bin/python"
    fi
fi

echo
echo "Using: $VENV_PYTHON"
"$VENV_PYTHON" --version

echo
echo "Checking project dependencies..."
if "$VENV_PYTHON" -c "import matplotlib, numpy, optionmc, pytest, scipy" >/dev/null 2>&1; then
    echo "Dependencies are already installed."
else
    echo "Installing project dependencies..."
    "$VENV_PYTHON" -m pip install -e ".[dev]"
fi

echo
echo "Running the automated test suite..."
"$VENV_PYTHON" -m pytest -q -p no:cacheprovider

echo
echo "Running Monte Carlo method comparison..."
"$VENV_PYTHON" scripts/run_comparison.py

echo
echo "Running parameter sensitivity analysis..."
"$VENV_PYTHON" scripts/run_sensitivity.py

echo
echo "Generated result files:"
if [[ -d "artifacts" ]]; then
    find artifacts -maxdepth 1 -type f -print | sort
else
    echo "No artifacts directory was created." >&2
    exit 1
fi

echo
echo "All OptionMC checks completed successfully."
