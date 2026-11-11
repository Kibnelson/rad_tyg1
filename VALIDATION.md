# Validation

The commands below record earlier checks. Current runbook commands use 30
worker threads; these earlier tests used four.


The bundled engine was built from source on Linux x86-64 with Python 3.10.22,
AVX-512 and four build jobs. Setup's native checks compare all 113 possible
matching-position counts against the saved raw-MinHash distance rule, check an
identity search, and save/reload the native graph. All passed.

```bash
PYTHON=python3.10 JOBS=4 bash setup.sh
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 4 --batch-size 100 --partition-capacity 1000
```

The 320-row fixture produced 220 insertions, 94 index-search drops and six
in-batch drops in one run. These counts sum to 320. HNSW is approximate; the
fixture does not require it to find every repeated document, and parallel graph
construction can change individual outcomes.

Checks also confirmed 320 ordered evidence rows, complete top40 lists after the
first batch, matching saved graph/ID counts, and that searches only see earlier
batches. A 125-document prefix exercised a partial final batch; reusing an
existing output directory was rejected without changing its saved summary.

Fixture SHA256:
`e21cb1c105592c5eb55f8f3e246a2dff786816adbc7f625aac7e14e7bbad4bbb`.
The 1,127 source-file hashes in [PROVENANCE.json](PROVENANCE.json) were checked.
These are functional checks, not throughput, exact-Jaccard or recall results.
