# Local native validation

Fresh `setup.sh` completed with Python 3.10.22 on Linux x86-64. It compiled both
vendored SIMD modules and checked duplicate detection with the actual native
library; no substitute engine was used.

Streaming checks verified document accounting and one decision per input row.
Local checks use the resources available on the validation host. The runbooks
use 30 worker threads on a VM with enough available CPUs.

Fixture SHA256: `e21cb1c105592c5eb55f8f3e246a2dff786816adbc7f625aac7e14e7bbad4bbb`.

```bash
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100 --partition-capacity 1000
```

Generated binaries, local environments and result files are not committed.
