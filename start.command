#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Set up the Python virtual environment first. See README.md."
  exit 1
fi
exec .venv/bin/python -m vmd
