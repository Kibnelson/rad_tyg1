# Faiss(Jaccard): raw-MinHash HNSW

This branch checks incoming documents against documents already kept, using
the MH Jaccard engine in modified Faiss. It filters duplicates within each
batch, searches one growing HNSW graph, then inserts the accepted documents.

Start with the [runbook](RUNBOOK.md). The included 320-document test dataset
needs no AWS account. The runbook then shows how to download and test more
FineWeb documents. [DATASETS.md](DATASETS.md) also covers S3 and Common Crawl.

The public configuration uses **M96, corpus/query efConstruction128,
efSearch150, top-k4 and distance cutoff30**. There is no
Bloom routing. The engine stores 112 uint32 MinHash values as 448 big-endian
bytes and compares matching signature positions, rather than RAD bitmaps.

Outputs include `summary.json`, per-batch timing/counts, accepted IDs, the final
native graph and **all ordered top-k4 candidates before admission** in `topk.jsonl`.

[PROVENANCE.json](PROVENANCE.json) records the bundled native engine and encoder.
The streaming runner checks every document.

Functional installation checks: [VALIDATION.md](VALIDATION.md).

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.

Run commands and the runner default use **30 worker threads**. Use a VM with
at least 30 available logical CPUs (32 vCPUs recommended).

For reproduction, use the same **GCP C3D high-memory machine: AMD Genoa,
32 CPU cores and 480 GB RAM**, with **30 workers**. See the
[experiment machine configuration](MACHINE_SPECS.md#gcp-machine-used-for-the-experiments).
