#!/usr/bin/env bash

python -m build  # creates sdist (.tar.gz) + wheel (.whl) in ./dist/
python -m twine check dist/*  # optional sanity check
