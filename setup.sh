#!/usr/bin/env bash
# Create the two Python environments:
#   .venv     neural-amp-modeler 0.13 (reads A2)
#   .venv-a1  neural-amp-modeler 0.11 (reads/writes A1, model version 0.5.4)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      PY="$c"; break
    fi
  done
fi
[[ -n "$PY" ]] || { echo "Python 3.10+ required (set PYTHON=...)" >&2; exit 1; }
echo "Using $PY ($("$PY" --version))"

"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-a2.txt

"$PY" -m venv .venv-a1
.venv-a1/bin/pip install -q --upgrade pip
.venv-a1/bin/pip install -q -r requirements-a1.txt

chmod +x nam-a2-to-a1
echo "Done. Try: ./nam-a2-to-a1 your-capture.nam"
