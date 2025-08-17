#!/usr/bin/env bash

set -ex

SCRIPT_PATH="${BASH_SOURCE:-$0}"
SCRIPT_DIR="$(dirname "$(realpath "$SCRIPT_PATH")")"

cd "$SCRIPT_DIR"
cd ..

export UV_PYTHON="$(python -c 'import sys; print(sys.executable)')"

uv pip install \
  "torch==2.5.0+cpu" "torchvision==0.20.0+cpu" \
  --index-url https://download.pytorch.org/whl/cpu

# dev install
uv pip install -e ".[dev]"

# Linux CPU torch when needed:
uv pip install ".[torch-cpu]" -c conda/cpu-constraints.txt

# optional extras
uv pip install ".[base]"
