# Source provenance

[PROVENANCE.json](PROVENANCE.json) records SHA256 hashes for the bundled
MinHash encoder and two native SIMD modules.

SIMD-DPK reimplements IBM Data Prep Kit's MinHash/LSH fuzzy-deduplication
algorithm as a standalone C++ SIMD engine. The full-reference engine checks
incoming documents against all retained signatures. The IBM DPK package and
its Ray pipeline are not bundled. The JSONL runner processes every input in
batches and writes one decision per document.

MurmurHash3 retains its public-domain notice in `licenses/`. The SIMD and
encoder files retain their original source headers; no additional third-party
license is inferred for them. See [NOTICE.md](NOTICE.md) and the preserved
native notices.
