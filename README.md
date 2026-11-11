# Faiss(Jaccard): raw-MinHash HNSW

This branch runs the MH Jaccard engine: 112 uint32 MinHash values are stored as
448 big-endian bytes, and the modified native Faiss metric compares matching
signature positions. It is **not** RAD's occupancy-bitmap distance.

Defaults match the completed middle-budget MH profile: **M96, corpus/query
efConstruction384, efSearch400, top40 and distance cutoff30**. A single graph
grows from empty. Each full-text batch is filtered in-batch, searched, and then
accepted documents are inserted. There is no Bloom routing.

Start with the [runbook](RUNBOOK.md). The included fixture tests the engine;
it is not paper-quality throughput or recall evidence.
See [DATASETS.md](DATASETS.md) for S3 inputs and public Common Crawl/FineWeb.

Outputs include `summary.json`, per-batch timing/counts, accepted IDs, the final
native graph and **all ordered top40 results before admission** in `topk.jsonl`.
These permit offline first4-versus-all40 inspection of the same search result;
they do not simulate an independent top-k4 search. Real-data recall and exact
literal-text Jaccard are not measured.

[PROVENANCE.json](PROVENANCE.json) binds the unchanged native engine and encoder.
This new portable all-row driver does not reproduce the earlier prepared
900K-corpus/100K-query workload or its numerical results.

Functional installation checks: [VALIDATION.md](VALIDATION.md).
