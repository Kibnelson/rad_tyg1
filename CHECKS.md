# Check your installation

After completing the build in [RUNBOOK.md](RUNBOOK.md), run from the cloned
system folder on compatible hardware:

```bash
.venv/bin/python scripts/check_fixture.py
.venv/bin/python scripts/check_milvus_sources.py
bash scripts/services.sh up
.venv/bin/python scripts/run.py --input data/sample.jsonl \
  --output results/checks --threads 30 --batch-size 100 --partition-capacity 1000
cat results/checks/SUMMARY.json
```

The source check verifies bundled files against their SHA256 inventory. The
native sample should finish with `status: completed`; inserted documents and
the two drop counters must account for all screened inputs. Check the output
schema and timing scope in [README.md](README.md#saved-files).

Choose a new output folder for another run. After testing:

```bash
bash scripts/services.sh down
```

This stops the supporting services while retaining their local data.
