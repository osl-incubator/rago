#!/usr/bin/env bash

set -ex

export UV_PYTHON="$(python -c 'import sys; print(sys.executable)')"

# dev install
uv pip install -e ".[dev]"

# Linux CPU torch when needed:
uv pip install ".[torch]"

# optional extras
uv pip install ".[base]"
