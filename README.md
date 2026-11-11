# RAD

RAD reads documents in batches, checks them against the documents already kept,
and inserts those that pass its duplicate checks. Start with the small included
sample; no AWS account is needed.

Follow [RUNBOOK.md](RUNBOOK.md) to install RAD, run the sample, and try a larger
input. [DATASETS.md](DATASETS.md) covers S3, FineWeb and Common Crawl downloads.

## How this version works

RAD uses MinHash bitmaps and HNSW indexes. Documents fill one partition at a time;
a full partition stays searchable while new documents enter the next partition.
Each partition has a Bloom filter that selects which queries need to search it.
All partitions stay in memory, and selected partitions are searched one at a time.

Defaults: **M96, corpus/query efConstruction96, efSearch100, top4, bitmap cutoff
D≤50, 10K-document batches and 1M kept documents per partition**. Routing uses
37 position-tagged groups of three MinHash values. Queries routed to no partition
are inserted. D≤50 is an approximate bitmap rule, not an exact Jaccard≥0.70 test.

The runbook uses `cache_scalar`, which needs AVX2 and POPCNT. The benchmark
kernel `cache_simd` additionally requires AVX-512 VPOPCNTDQ; select it explicitly
only on a compatible CPU. The chosen kernel is recorded in the results.
The earlier graph profile is available with
`--ef-construction 128 --ef-search 150 --top-k 40`.

## Results

`summary.json` shows document counts, settings, and total documents divided by
processing time. `batches.csv` gives counts, stage timings and partition visits
for each batch. `evidence/batch_*.jsonl` records ordered candidates and decisions;
`batch_*_metadata.json` gives per-partition search details. Use `--top-k 40` to
keep all returned top40 candidates for later first4-versus-top40 inspection.
Index IDs are input positions; `source_id` keeps the supplied document ID.

Processing time includes input reading, MinHash, in-batch filtering, bitmap
creation, routing/search/merge, duplicate decisions and insertion. Writing
results/checkpoints and setup/download/build time are excluded. There are no
separate warmup searches in this public runner. Sample timings are installation
checks, not the earlier paper results; real-data recall is not measured.

[Native test results](VALIDATION.json) and [source provenance](PROVENANCE.md)
record what was checked and where the code came from.

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.
