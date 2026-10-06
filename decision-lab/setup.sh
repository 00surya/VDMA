#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check --cache-dir .cache/pip -r requirements.lock
exec .venv/bin/python download_model.py
