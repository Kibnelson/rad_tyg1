#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RAD_PYTHON="${PYTHON:-python3}"
"$RAD_PYTHON" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10, 3.11 or 3.12 (set PYTHON=python3.12)."'
if [ ! -x .venv/bin/python ]; then "$RAD_PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements-rad.txt
.venv/bin/python scripts/setup_native.py --jobs "${JOBS:-4}"
