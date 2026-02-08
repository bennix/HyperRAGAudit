#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Create venv with Python 3.13 if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (Python 3.13)..."
    python3.13 -m venv .venv
    echo "Installing dependencies..."
    .venv/bin/pip install -r requirements.txt
fi

echo "Starting HyperRAG Audit..."
exec .venv/bin/streamlit run app.py "$@"
