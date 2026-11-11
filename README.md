# SIMD-DPK: native full-reference deduplication

This branch runs the SIMD full-reference baseline used in our experiments.
Every document is screened in order. It computes 112 MinHash values, applies
the original 14×8 in-batch filter, then runs the original native SIMD algorithm
over the retained reference plus the current survivors, using threshold 0.7.
Accepted documents extend the reference. No ranked top-k or HNSW index is used.
The threshold is applied to matching MinHash positions, not exact text Jaccard.

Start with the [runbook](RUNBOOK.md). The included 320-document fixture tests
installation and insert/drop handling; it is not a benchmark dataset.
See [DATASETS.md](DATASETS.md) for our S3 inputs and public Common Crawl/FineWeb.

Outputs: `summary.json`, per-batch counts/timings, one decision per source row,
accepted IDs and retained signatures. Real-data recall and exact Jaccard are
not measured. This is the native SIMD baseline, not an IBM DPK/Ray ground-truth
job. Its full-reference rebuilding cost is included in processing time.

The native files and MinHash encoder are unchanged; [PROVENANCE.json](PROVENANCE.json)
records their source revisions and hashes. This portable all-row driver is new;
it does not reproduce the earlier 900K-corpus/100K-query prepared workload.

Functional installation checks: [VALIDATION.md](VALIDATION.md).
