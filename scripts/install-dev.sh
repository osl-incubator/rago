#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
SCRIPT_PATH="${BASH_SOURCE:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR/.."

# Make uv install into the current interpreter (conda/venv)
export UV_PYTHON="$(python -c 'import sys; print(sys.executable)')"

OS="$(uname -s || echo unknown)"

case "$OS" in
  Linux*)
    # Linux needs explicit CPU wheels and the CPU index
    uv pip install \
      "torch==2.5.0+cpu" "torchvision==0.20.0+cpu" \
      --index-url https://download.pytorch.org/whl/cpu
    ;;

  Darwin*)
    # macOS PyPI wheels are CPU-only by default
    ;;

  MINGW*|MSYS*|CYGWIN*)
    # Windows (Git Bash/MSYS). PyPI wheels are CPU-only by default.
    ;;

  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

uv pip install -e ".[dev,all]"
