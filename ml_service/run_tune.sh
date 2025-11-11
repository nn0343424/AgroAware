#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run tune_hparams.py inside the project's venv
# Usage: ./run_tune.sh [args...]    (args forwarded to tune_hparams.py)

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYBIN="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
  echo ".venv not found. Running setup..."
  "$ROOT_DIR/setup_env.sh"
fi

echo "Running tune_hparams.py with $PYBIN"
exec "$PYBIN" "$ROOT_DIR/tune_hparams.py" "$@"
