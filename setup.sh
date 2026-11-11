#!/usr/bin/env bash
set -euo pipefail
PUBLIC_ENGINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ENGINE_PYTHON="${PYTHON:-python3}"
"$PUBLIC_ENGINE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12 (for example PYTHON=python3.12 bash setup.sh)."'
"$PUBLIC_ENGINE_PYTHON" -m venv "$PUBLIC_ENGINE_ROOT/.venv"
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" -m pip install -r "$PUBLIC_ENGINE_ROOT/requirements.txt"
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" "$PUBLIC_ENGINE_ROOT/scripts/build_native.py"
printf "%s\n" "Setup complete. Use .venv/bin/python scripts/run.py --help. Example: see RUNBOOK.md."
