#!/usr/bin/env bash
# Target VM only: start the locally built standalone server in the foreground.
set -euo pipefail
RADV2_SERVER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RADV2_SERVER_CONFIG="${1:-$RADV2_SERVER_ROOT/runs/milvus-physical-1m/config}"
RADV2_SERVER_THREADS="${2:-28}"
RADV2_INSTANCE_SLOT="${3:-}"
RADV2_EXPLICIT_CPUS="${4:-}"
case "$RADV2_SERVER_THREADS" in ''|*[!0-9]*|0) printf '%s\n' 'Use a positive CPU budget.' >&2; exit 1 ;; esac
RADV2_CONFIG_ARGS=(--prepare-config "$RADV2_SERVER_CONFIG")
if [ -n "$RADV2_INSTANCE_SLOT" ]; then RADV2_CONFIG_ARGS+=(--instance-slot "$RADV2_INSTANCE_SLOT"); fi
python3 "$RADV2_SERVER_ROOT/scripts/check_milvus_sources.py" "${RADV2_CONFIG_ARGS[@]}"
RADV2_CPU_LIST="$(python3 - "$RADV2_SERVER_ROOT" "$RADV2_SERVER_THREADS" "$RADV2_EXPLICIT_CPUS" <<'PY'
import hashlib,json,os
from pathlib import Path
import sys
root,workers=Path(sys.argv[1]),int(sys.argv[2])
build=json.loads((root/'build/milvus_build.json').read_text())
binary=root/'external/milvus/bin/milvus'
if (build['status']!='built' or build['binary']!=str(binary)
    or build['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest()
    or build['lock_sha256']!=hashlib.sha256((root/'milvus_sources.lock.json').read_bytes()).hexdigest()):
    raise SystemExit('STOP: build/source receipt differs; rebuild the owned sources.')
for name,checksum in build['libraries'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=checksum:
        raise SystemExit('STOP: a native library changed; rebuild the owned sources.')
cpus=sorted(os.sched_getaffinity(0))
if workers>len(cpus):raise SystemExit('STOP: CPU budget exceeds available affinity.')
selected=list(map(int,sys.argv[3].split(','))) if sys.argv[3] else cpus[:workers]
if len(selected)!=workers or len(set(selected))!=workers or not set(selected)<=set(cpus):
    raise SystemExit('STOP: explicit CPU list must contain exactly the requested available workers.')
print(','.join(map(str,selected)))
PY
)"
export MILVUSCONF="$(realpath "$RADV2_SERVER_CONFIG")"
export GOMAXPROCS="$RADV2_SERVER_THREADS" OMP_NUM_THREADS="$RADV2_SERVER_THREADS"
if [ -n "$RADV2_INSTANCE_SLOT" ]; then
  export METRICS_PORT="$((9091 + RADV2_INSTANCE_SLOT * 100))"
fi
cd "$RADV2_SERVER_ROOT/external/milvus"
source scripts/setenv.sh
# Upstream setenv.sh disables errexit; restore our launch checks afterward.
set -euo pipefail
printf 'Starting OWNED Milvus: physical row cap=1000000; CPUs=%s; config=%s\n' "$RADV2_CPU_LIST" "$MILVUSCONF"
RADV2_INSTANCE_ALIAS="radv2-$(printf '%s' "$MILVUSCONF" | sha256sum | cut -c1-16)"
exec taskset -c "$RADV2_CPU_LIST" bin/milvus run standalone -alias "$RADV2_INSTANCE_ALIAS"
