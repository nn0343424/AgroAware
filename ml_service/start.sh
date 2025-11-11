#!/usr/bin/env bash
set -euo pipefail

# Start script for ml_service
# Usage: ./start.sh [--port PORT]

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYBIN="${VENV_DIR}/bin/python"
UVICORN_MODULE="uvicorn"
APP_MODULE="api:app"
PORT="8000"
HOST="0.0.0.0"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo "Usage: ./start.sh [PORT]"
  exit 0
fi

if [ "$#" -ge 1 ]; then
  PORT="$1"
fi

# Ensure environment is ready
if [ ! -d "$VENV_DIR" ]; then
  echo ".venv not found, running setup..."
  ./setup_env.sh
fi

# Run uvicorn via the venv python so it's isolated
echo "Starting ml_service on http://$HOST:$PORT"
exec "$PYBIN" -m "$UVICORN_MODULE" "$APP_MODULE" --reload --host "$HOST" --port "$PORT"
