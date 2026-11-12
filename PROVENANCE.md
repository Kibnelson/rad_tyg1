# Source provenance

The bundled source includes modified Faiss, the SIMD batch filter, native
bitmap/Bloom helpers and the partitioned index modules. [PROVENANCE.json](PROVENANCE.json)
lists these components and their public settings.

The JSONL runner processes full-text documents in batches and writes candidate
evidence, stage timings and local checkpoints. It uses M96, corpus/query
efConstruction128, efSearch150, top-k4 and bitmap cutoff30, with a nominal
bitmap similarity target of 0.70. Bitmap distance is serialized as
`D = floor(100 × (1 − bitmap Jaccard) + 1e-6)`; this is not an exact text-Jaccard
threshold. The runner makes no separate warmup searches.

Faiss's MIT license and third-party notices are preserved in
[external/faiss/LICENSE](external/faiss/LICENSE) and
[external/faiss/THIRD_PARTY_NOTICES](external/faiss/THIRD_PARTY_NOTICES).
MurmurHash3 retains its public-domain notice in `external/simd/MurmurHash3.h`.
See [NOTICE.md](NOTICE.md) for source and data attribution.
