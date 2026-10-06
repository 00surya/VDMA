#!/bin/zsh
cd -- "${0:A:h:h}" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
.venv/bin/python scripts/start_mobile.py
result=$?
if (( result != 0 )); then
  read -r '?Press Return to close.'
fi
exit "$result"
