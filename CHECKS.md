# Check your installation

Setup checks the native MinHash distance rule, identity search and graph
save/reload. After completing [RUNBOOK.md](RUNBOOK.md), run the included input:

```bash
.venv/bin/python scripts/check_fixture.py
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100 --partition-capacity 1000 \
  --ef-construction 384 --ef-search 400 --top-k 4
cat results/sample/summary.json
```

The fixture check verifies the input hashes and row counts. Check that inserted,
index-search drops and in-batch drops account for all screened inputs. The run
also saves ordered candidates, accepted IDs and the native graph. HNSW is
approximate, so individual insert/drop totals are not fixed expected results.

Use a new output folder for another run. See [README.md](README.md) for output
files and [MACHINE_SPECS.md](MACHINE_SPECS.md) for the required CPU instructions.
