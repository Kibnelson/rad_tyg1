#!/usr/bin/env bash
# Build the included Milvus, Knowhere, SDK and SIMD sources; start no services.
set -euo pipefail
PUBLIC_MILVUS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_MILVUS_PYTHON="${PYTHON_BIN:-python3.12}"
PUBLIC_MILVUS_JOBS="${JOBS:-2}"
cd "$PUBLIC_MILVUS_ROOT"

check_python_build() {
  if ! command -v "$PUBLIC_MILVUS_PYTHON" >/dev/null 2>&1; then
    printf 'Selected interpreter is unavailable: %s\n' "$PUBLIC_MILVUS_PYTHON" >&2
    return 1
  fi
  "$PUBLIC_MILVUS_PYTHON" - <<'PY'
import sys
import sysconfig
from pathlib import Path
if not (3,10) <= sys.version_info[:2] <= (3,12):
    raise SystemExit('Use Python 3.10–3.12')
minor=f'{sys.version_info.major}.{sys.version_info.minor}'
try:
    import ensurepip
    import venv
except ImportError:
    raise SystemExit(f'Install python{minor}-venv for the selected interpreter')
header=Path(sysconfig.get_path('include'))/'Python.h'
if not header.is_file():
    raise SystemExit(f'Install python{minor}-dev: missing {header}')
print(f'Python {minor}: venv/ensurepip and matching Python.h available')
PY
}

if [ "${1:-}" = --check-python ] && [ "$#" -eq 1 ]; then
  check_python_build
  exit
elif [ "$#" -ne 0 ]; then
  printf '%s\n' 'Use: bash scripts/setup.sh [--check-python]' >&2
  exit 2
fi

sudo apt-get update
# A minimal Ubuntu VM may not have the selected interpreter installed yet.
if ! command -v "$PUBLIC_MILVUS_PYTHON" >/dev/null 2>&1; then
  case "$PUBLIC_MILVUS_PYTHON" in
    python3.10|python3.11|python3.12|/usr/bin/python3.10|/usr/bin/python3.11|/usr/bin/python3.12)
      PUBLIC_MILVUS_PYTHON_PACKAGE="${PUBLIC_MILVUS_PYTHON##*/}"
      sudo apt-get install -y "$PUBLIC_MILVUS_PYTHON_PACKAGE" \
        "$PUBLIC_MILVUS_PYTHON_PACKAGE-venv" "$PUBLIC_MILVUS_PYTHON_PACKAGE-dev"
      ;;
    *) printf 'Install the selected interpreter first: %s\n' "$PUBLIC_MILVUS_PYTHON" >&2; exit 1 ;;
  esac
fi
PUBLIC_MILVUS_MINOR="$("$PUBLIC_MILVUS_PYTHON" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10–3.12"; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! check_python_build >/dev/null 2>&1; then
  sudo apt-get install -y "python$PUBLIC_MILVUS_MINOR-venv" "python$PUBLIC_MILVUS_MINOR-dev"
fi
check_python_build
sudo apt-get install -y git g++ gcc gfortran make ninja-build ccache libssl-dev zlib1g-dev \
  zip unzip curl ca-certificates pkg-config uuid-dev libaio-dev libgoogle-perftools-dev \
  libtool m4 autoconf automake lcov libopenblas-dev liblapack-dev
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get install -y docker.io docker-compose-v2
fi
sudo docker compose version
"$PUBLIC_MILVUS_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
SETUPTOOLS_SCM_PRETEND_VERSION=3.0.2 .venv/bin/python -m pip install ./external/pymilvus
.venv/bin/python scripts/bootstrap_toolchains.py
source .build/environment.sh
.venv/bin/python scripts/build_simd.py
bash scripts/build_milvus_local.sh "$PUBLIC_MILVUS_JOBS"
printf '%s\n' 'Build complete. Run the sample command in RUNBOOK.md.'
