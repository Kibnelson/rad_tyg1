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
Selected partitions are searched one at a time. At the design level, partition
residency can follow the memory budget: keep all partitions in RAM, or use a
smaller working set, such as three, and load others as needed. The supplied
runner uses the all-resident policy.

Defaults: **M96, corpus/query efConstruction128, efSearch150, top4, bitmap cutoff
D≤30, 10K-document batches and 1M kept documents per partition**. Routing uses
37 position-tagged groups of three MinHash values. Queries routed to no partition
are inserted. The bitmap similarity target is 0.70. Native distance is serialized
as `D = floor(100 × (1 − bitmap Jaccard) + 1e-6)`, and admission uses `D≤30`.
This quantized bitmap rule is not an exact text-Jaccard threshold.

The runbook passes `--distance-mode cache_scalar`, which needs AVX2 and POPCNT.
The runner defaults to `cache_simd` if that option is omitted; this kernel
additionally requires AVX-512 VPOPCNTDQ. The chosen kernel is recorded in the results.

## Saved files

`summary.json` shows document counts, settings, and total documents divided by
processing time. `batches.csv` gives counts, stage timings and partition visits
for each batch. `evidence/batch_*.jsonl` records ordered candidates and decisions;
`batch_*_metadata.json` gives per-partition search details. Each searched query
keeps up to four ordered candidates.
Index IDs are input positions; `source_id` keeps the supplied document ID.

Processing time includes input reading, MinHash, in-batch filtering, bitmap
creation, routing/search/merge, duplicate decisions and insertion. Writing
results/checkpoints and setup/download/build time are excluded. There are no
separate warmup searches in this public runner.

[CHECKS.md](CHECKS.md) explains how to check the installation.
[PROVENANCE.md](PROVENANCE.md) describes the bundled source and licenses.

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.

Run commands and the runner default use **30 worker threads**. Use a VM with
at least 30 available logical CPUs (32 vCPUs recommended).

For reproduction, use the same **GCP C3D high-memory machine: AMD Genoa,
32 CPU cores and 480 GB RAM**, with **30 workers**. See the
[reference GCP configuration](MACHINE_SPECS.md#reference-gcp-setup).
