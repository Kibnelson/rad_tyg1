# Run RAD on another input

Complete the README setup first. Inputs are JSONL records with `text` and an
optional original `id`; input order and full text are preserved. Download a
bounded prefix from the prepared S3 data (requires normal AWS read access):

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py s3 --dataset cc_main --rows 10000 --output data/cc_main.jsonl
```

Without AWS credentials, use public FineWeb or Common Crawl instead:

```bash
.venv/bin/python scripts/get_data.py fineweb --rows 10000 --output data/fineweb.jsonl
.venv/bin/python scripts/get_data.py common-crawl --crawl CC-MAIN-2024-10 --rows 10000 --output data/common_crawl.jsonl
```

Then run on the selected JSONL file:

```bash
.venv/bin/python scripts/run.py --input data/cc_main.jsonl --output results/cc_main \
  --threads 28 --distance-mode cache_simd
```

Defaults: **M96, corpus/query efConstruction96, efSearch100, top-k4, bitmap
cutoff D≤50, 10K batches and 1M retained documents per partition**. This is the
lower-search-budget Common Crawl profile. The earlier profile is available
with `--ef-construction 128 --ef-search 150 --top-k 40`. Other useful options:
`--max-documents 100000`, `--threads 4`, and `--distance-mode cache_scalar`.
D≤50 is a bitmap rule; it is not an exact Jaccard≥0.70 guarantee.

## What the runner does

Every document is screened. It computes 112-value MinHash, applies the native
in-batch deduplicator, and converts survivors to bitmaps. Bloom routing uses
37 position-tagged groups of three MinHash values. The main search visits
selected partitions sequentially and merges top-k results before D≤50 admission.
Zero-route queries are admitted. Accepted documents are appended only after
the batch's search, updating the active graph and its Bloom filter. Full
partitions remain searchable and all partitions remain in memory.

## Results and restart

`summary.json` contains counts, parameters and pooled processing throughput;
`batches.csv` has additive per-batch timings and routing visits; each
`evidence/batch_*_metadata.json` retains native per-partition search diagnostics. Ordered native
candidates and decisions are in `evidence/batch_*.jsonl`; changing `--top-k 40`
preserves all returned top40 candidates for offline first4 analysis. IDs in the index are zero-based
input ordinals; each observation also retains `source_id`. Recall is not measured.
Evidence/checkpoint/report writes are outside reported processing throughput;
setup/download/build time is not query time. This public runner makes no
separate warmup searches and does not claim to reproduce prior figure QPS.

To continue after Ctrl+C, repeat the same command with `--resume`. It reloads
the last committed graph/Bloom checkpoint and checks the input hash and run
configuration. Use a new output directory to change any graph or batch setting.
