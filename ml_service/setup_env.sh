#!/usr/bin/env bash
set -euo pipefail

# Idempotent environment setup for ml_service
# Usage: ./setup_env.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
PYBIN="${VENV_DIR}/bin/python"
PIPBIN="${VENV_DIR}/bin/pip"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Upgrade pip and setuptools
"$PYBIN" -m pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "$REQ_FILE" ]; then
  echo "Installing dependencies from $REQ_FILE"
  "$PIPBIN" install -r "$REQ_FILE"
else
  echo "No requirements.txt found at $REQ_FILE - skipping"
fi

echo "Environment ready. Activate with: source $VENV_DIR/bin/activate"
