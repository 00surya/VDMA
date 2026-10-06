#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo 'This experiment uses MLX and requires an Apple Silicon Mac.' >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check --cache-dir .cache/pip -r requirements.lock
.venv/bin/python download_model.py
