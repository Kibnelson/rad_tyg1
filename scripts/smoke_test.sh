#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RAD_SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rad-native-smoke.XXXXXXXX")"
RAD_SMOKE_MODE="${RAD_DISTANCE_MODE:-cache_scalar}"
.venv/bin/python scripts/run.py --input data/sample.jsonl --output "$RAD_SMOKE_DIR/run" \
  --threads 2 --batch-size 100 --partition-capacity 32 --distance-mode "$RAD_SMOKE_MODE"
.venv/bin/python scripts/run.py --input data/sample.jsonl --output "$RAD_SMOKE_DIR/run" \
  --threads 2 --batch-size 100 --partition-capacity 32 --distance-mode "$RAD_SMOKE_MODE" --resume
.venv/bin/python - "$RAD_SMOKE_DIR/run" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text())
assert s['status']=='completed' and s['screened']==320
assert s['screened']==sum(s[k] for k in ('inserted','index_search_drops','in_batch_dedup_drops'))
assert s['inserted']==s['stored_rows']==sum(s['partition_counts'])
assert len(s['partition_counts'])>1 and all(n==32 for n in s['partition_counts'][:-1])
rows=[json.loads(line) for f in sorted((p/'evidence').glob('batch_*.jsonl')) for line in f.read_text().splitlines()]
assert len(rows)==320 and sum(r['inserted'] for r in rows)==s['inserted']
assert any(r['candidates'] for r in rows)
print('NATIVE RAD SMOKE PASSED: counts, raw evidence, partition rollover and checkpoint reload.')
print('Test output:',p)
PY
