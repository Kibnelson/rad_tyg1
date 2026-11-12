# Source provenance

[PROVENANCE.json](PROVENANCE.json) records SHA256 hashes for the bundled
encoder, SIMD batch filter and modified Faiss sources.

The Faiss engine compares raw MinHash signatures, using its MinHash Jaccard
distance implementation. It includes the MIT license and third-party notices
in `vendor/faiss/`. The JSONL runner screens every input document in batches.
Its configuration uses M96, corpus/query efConstruction384, efSearch400,
top-k4 and distance cutoff30.

MurmurHash3 retains its public-domain notice in `licenses/`. The SIMD and
encoder files retain their original source headers; no additional third-party
license is inferred for them. See [NOTICE.md](NOTICE.md) and the preserved
native notices.
