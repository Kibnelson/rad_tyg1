# SIMD-DPK: native full-reference deduplication

This branch checks incoming documents against documents already kept, then
adds those it does not identify as duplicates. Every input document is checked
in order, using the original SIMD-DPK full-reference engine.

Start with the [runbook](RUNBOOK.md). The included 320-document test dataset
needs no AWS account. The runbook then shows how to download and test more
FineWeb documents. [DATASETS.md](DATASETS.md) also covers S3 and Common Crawl.

The engine uses 112 MinHash values, a 14×8 in-batch filter and threshold 0.7
over the complete retained reference plus the current batch's survivors.
This threshold compares MinHash positions; it is not exact text Jaccard.
There is no ranked top-k limit or HNSW index.

Outputs: `summary.json`, per-batch records, one decision per source row,
accepted IDs and retained signatures. Each batch checks the complete retained
reference before adding accepted documents.

The native files and MinHash encoder are unchanged; [PROVENANCE.json](PROVENANCE.json)
records their source revisions and hashes. The streaming runner checks every
document before insertion.

Functional installation checks: [VALIDATION.md](VALIDATION.md).

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.

Run commands and the runner default use **30 worker threads**. Use a VM with
at least 30 available logical CPUs (32 vCPUs recommended).
