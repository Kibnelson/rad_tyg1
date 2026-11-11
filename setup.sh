#!/usr/bin/env bash
set -euo pipefail
PUBLIC_ENGINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ENGINE_PYTHON="${PYTHON:-python3}"
"$PUBLIC_ENGINE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12 (for example PYTHON=python3.12 bash setup.sh)."'
"$PUBLIC_ENGINE_PYTHON" -m venv "$PUBLIC_ENGINE_ROOT/.venv"
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" -m pip install -r "$PUBLIC_ENGINE_ROOT/requirements.txt"
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" "$PUBLIC_ENGINE_ROOT/scripts/build_native.py"
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" -c 'from pathlib import Path; flags=set(Path("/proc/cpuinfo").read_text().split()); required={"avx2","avx512f","avx512cd","avx512dq","avx512bw","avx512vl","f16c","fma","popcnt","bmi2"}; assert required <= flags, "Native Faiss build requires CPU flags: " + ", ".join(sorted(required-flags))'
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS:-4}"
export PATH="$PUBLIC_ENGINE_ROOT/.venv/bin:$PATH"
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" -m pip install --no-build-isolation --no-deps "$PUBLIC_ENGINE_ROOT/vendor/faiss" \
  --config-settings=build-dir="$PUBLIC_ENGINE_ROOT/build/faiss" \
  --config-settings=cmake.define.SWIG_EXECUTABLE="$PUBLIC_ENGINE_ROOT/.venv/bin/swig" \
  --config-settings=cmake.define.FAISS_OPT_LEVEL=avx512 \
  --config-settings=cmake.define.FAISS_USE_LTO=OFF \
  --config-settings=cmake.define.BUILD_TESTING=OFF
"$PUBLIC_ENGINE_ROOT/.venv/bin/python" "$PUBLIC_ENGINE_ROOT/scripts/check_faiss.py"
printf "%s\n" "Setup complete. Use .venv/bin/python scripts/run.py --help. Example: see RUNBOOK.md."
