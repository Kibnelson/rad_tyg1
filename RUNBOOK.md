# Run Milvus afresh

Use an x86-64 Ubuntu VM with sudo and AVX-512F/DQ/VL/BW, AVX2 and POPCNT. Ubuntu 24.04 with Python 3.12 is the reference environment. Setup installs the selected Python interpreter, matching venv/development packages, and Docker Engine with Compose when absent. An existing Docker installation must provide Compose. Python 3.10–3.12 is supported; other Ubuntu releases need the selected version's packages available in their configured apt repositories. Allow at least 64 GB RAM and 100 GB free disk for the native build; the 30M datasets/results need additional disk. Build dependencies download from public upstream services.

```bash
git clone --branch milvus --single-branch https://github.com/Kibnelson/rad_tyg1.git
cd rad_tyg1
PYTHON_BIN=python3.12 JOBS=2 bash scripts/setup.sh
bash scripts/services.sh up
.venv/bin/python scripts/run.py --input data/sample.jsonl \
  --output results/sample --threads 4 --batch-size 100 --partition-capacity 1000
```

Use a fresh output directory for each run. The runner starts and stops only its own native server processes; etcd/MinIO remain running until:

```bash
bash scripts/services.sh down
```

For a larger full-text stream, first download input following [DATASETS.md](DATASETS.md), then:

```bash
.venv/bin/python scripts/run.py --input data/cc_main_download.jsonl \
  --output results/cc_main --threads 28 --batch-size 10000 --partition-capacity 1000000
```

Choose no more threads than the VM's available CPU affinity. `--max-documents 10000` limits a trial without truncating individual documents. Each batch is MinHash encoded, deduplicated in batch, searched against existing partitions, and its unmatched documents are appended. Full partitions are sealed; the final partial partition is sealed at run completion. This small public driver starts a new run; the original VM recovery/publication workflow is not exposed here.

The local profile permits native index building from one retained row, so the tiny fixture's final segment can be indexed. This lowers Milvus's default 1,024-row index-building threshold; it leaves the 14×8 LSH settings and retained partition capacity unchanged.

`batches.csv` records counts, stage times and `screened / processing_seconds`; `SUMMARY.json` reports pooled throughput and full local wall time. Processing includes MinHash, in-batch deduplication, native search, admission, and append/readiness. Creating native owners during append is included in insert time. Input iteration, raw-result writes, store-context setup, final partial-partition sealing and shutdown are outside that processing timer and inside full wall time. Build/download time is outside both. `raw/batch_*.jsonl` preserves ordered top4 candidates outside the processing timer. Stream IDs are zero-based input positions; source IDs are also retained. No raw input text is copied to the result JSONL.

Before sharing a timing result, use the same timer scope, thread budget, batch size and partition capacity. The tiny Git input validates the execution path; it is not a performance experiment.
