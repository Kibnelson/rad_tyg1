# Check your installation

After completing [RUNBOOK.md](RUNBOOK.md), run from the cloned system folder:

```bash
.venv/bin/python scripts/check_fixture.py
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100
cat results/sample/summary.json
```

The fixture check verifies the included input hashes and row counts. Check that
the native run completes, inserted and dropped counts account for every input,
and the evidence contains one decision per screened document. The runner uses
the full retained reference; it has no ranked top-k setting.

Use a new output folder for another run. Output files and timing scope are
described in [README.md](README.md).
