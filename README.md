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

Outputs: `summary.json`, per-batch counts/timings, one decision per source row,
accepted IDs and retained signatures. Real-data recall and exact Jaccard are
not measured. This is the native SIMD baseline, not an IBM DPK/Ray ground-truth
job. Its full-reference rebuilding cost is included in processing time.

The native files and MinHash encoder are unchanged; [PROVENANCE.json](PROVENANCE.json)
records their source revisions and hashes. The streaming runner is new and
checks every document. The small test does not reproduce paper results or the
earlier 900K-corpus/100K-query workload.

Functional installation checks: [VALIDATION.md](VALIDATION.md).

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.
