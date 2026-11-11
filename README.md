# Milvus MinHash-LSH

This branch contains the native Milvus/Knowhere implementation used for the full-text streaming runs. It screens every document, keeps ordered top4 candidates, and inserts documents with no native LSH candidate. It uses 112 uint32 MinHash values, 14 bands of 8 values, and sequential partitions of at most 1M **retained** documents. A source-million boundary does not seal a partition.

Follow [RUNBOOK.md](RUNBOOK.md) for a small test, then use [DATASETS.md](DATASETS.md) to obtain more input. No access to the original private repository is required. The Git sample contains authored examples, introduced repeats and short dataset excerpts; it is for testing, not for reproducing benchmark scores.

The included custom server is built from source. Docker runs its local etcd/MinIO dependencies. This is not the stock Milvus Docker image.

See [PROVENANCE.md](PROVENANCE.md) for source versions and licenses. Exact Jaccard and recall are not measured: an LSH candidate is an approximate match, and its returned score is a placeholder.

Functional installation checks: [VALIDATION.md](VALIDATION.md).
