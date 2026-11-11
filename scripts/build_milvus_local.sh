#!/usr/bin/env bash
# Target VM only: builds the owned source, without starting a server/experiment.
set -euo pipefail
RADV2_BUILD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RADV2_BUILD_JOBS="${1:-8}"
case "$RADV2_BUILD_JOBS" in ''|*[!0-9]*|0) printf '%s\n' 'Use a positive build job count.' >&2; exit 1 ;; esac
python3 "$RADV2_BUILD_ROOT/scripts/check_milvus_sources.py"
command -v go >/dev/null
command -v cmake >/dev/null
command -v make >/dev/null
mkdir -p "$RADV2_BUILD_ROOT/build"
RADV2_SOURCE_STATE="$RADV2_BUILD_ROOT/build/milvus_sources.json"
python3 "$RADV2_BUILD_ROOT/scripts/check_milvus_sources.py" --report "$RADV2_SOURCE_STATE"
RADV2_PROJECT_REV="$(git -C "$RADV2_BUILD_ROOT" rev-parse HEAD 2>/dev/null || sha256sum "$RADV2_BUILD_ROOT/milvus_sources.lock.json" | cut -c1-40)"
export jobs="$RADV2_BUILD_JOBS"
export GOMAXPROCS="$RADV2_BUILD_JOBS"
export CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-} -DRADV2_KNOWHERE_SOURCE=$RADV2_BUILD_ROOT/external/knowhere"
cd "$RADV2_BUILD_ROOT/external/milvus"
make milvus MILVUS_VERSION=3.0.5-radv2 GIT_COMMIT="$RADV2_PROJECT_REV" GIT_BRANCH=radv2-physical-segments
test -x bin/milvus
python3 "$RADV2_BUILD_ROOT/scripts/check_milvus_sources.py"
python3 - "$RADV2_BUILD_ROOT" <<'PY'
import hashlib,json
from pathlib import Path
import sys
root=Path(sys.argv[1])
binary=root/'external/milvus/bin/milvus'
value=json.loads((root/'build/milvus_sources.json').read_text())
value.update(status='built',scope='source_build_only',binary=str(binary),
             binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
output=root/'external/milvus/internal/core/output'
libraries=sorted({p.resolve() for pattern in ('libmilvus_core.so*','libknowhere.so*')
                  for p in output.rglob(pattern) if p.is_file()})
if not any('libmilvus_core.so' in p.name for p in libraries):
    raise SystemExit('STOP: native core shared library missing from owned output.')
value['libraries']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}
(root/'build/milvus_build.json').write_text(json.dumps(value,indent=2)+'\n')
print('LOCAL MILVUS BUILT:',binary,'; physical-segment runtime validation is still required.')
PY
