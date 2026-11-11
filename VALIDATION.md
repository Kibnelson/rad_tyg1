# Local native validation

The commands below record earlier checks. Current runbook commands use 30
worker threads; these earlier tests used four.


Fresh `setup.sh` completed with Python 3.10.22 on Linux x86-64. It compiled both
vendored SIMD modules and checked duplicate detection with the actual native
library; no substitute engine was used.

The documented sample command screened all **320 rows** in four batches:
**216 inserted, 98 full-reference drops, 6 in-batch drops**. Counts close exactly,
and the output contains one decision per input row. No recall is claimed.

Fixture SHA256: `e21cb1c105592c5eb55f8f3e246a2dff786816adbc7f625aac7e14e7bbad4bbb`.

```bash
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 4 --batch-size 100 --partition-capacity 1000
```

These are installation/function checks, not benchmark performance results.
Generated binaries, local environments and result files are not committed.
