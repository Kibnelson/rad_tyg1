# Faiss(Jaccard): raw-MinHash HNSW

This branch checks incoming documents against documents already kept, using
the MH Jaccard engine in modified Faiss. It filters duplicates within each
batch, searches one growing HNSW graph, then inserts the accepted documents.

Start with the [runbook](RUNBOOK.md). The included 320-document test dataset
needs no AWS account. The runbook then shows how to download and test more
FineWeb documents. [DATASETS.md](DATASETS.md) also covers S3 and Common Crawl.

Defaults match the completed middle-budget MH profile: **M96, corpus/query
efConstruction384, efSearch400, top40 and distance cutoff30**. There is no
Bloom routing. The engine stores 112 uint32 MinHash values as 448 big-endian
bytes and compares matching signature positions, rather than RAD bitmaps.

Outputs include `summary.json`, per-batch timing/counts, accepted IDs, the final
native graph and **all ordered top40 results before admission** in `topk.jsonl`.
These permit offline first4-versus-all40 inspection of the same search result;
they do not simulate an independent top-k4 search. Real-data recall and exact
literal-text Jaccard are not measured.

[PROVENANCE.json](PROVENANCE.json) binds the unchanged native engine and encoder.
The streaming runner checks every document. The small test does not reproduce
paper results or the earlier 900K-corpus/100K-query workload.

Functional installation checks: [VALIDATION.md](VALIDATION.md).

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.
