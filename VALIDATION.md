# Installation checks

Setup checks the raw-MinHash distance rule, identity search and native graph
save/reload. The qualification graph uses M96, efConstruction128, efSearch150
and top-k4.

After setup, use the included input to check the complete streaming path:

```bash
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100 --partition-capacity 1000
```

The runner checks that every input document is either inserted, rejected by
index search or filtered within its batch. It saves one ordered evidence row
per input document, accepted IDs and the final native graph. Use a new output
directory for each test.

The source file hashes are recorded in [PROVENANCE.json](PROVENANCE.json).
The input hash and source attribution are in
[data/SAMPLE_PROVENANCE.json](data/SAMPLE_PROVENANCE.json).
