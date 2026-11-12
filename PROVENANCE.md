# Source provenance

Research snapshot: `a4b78aaee37dea08662c36d97170f0e45cdbce7a`.
[Distributed-file SHA256 values](PROVENANCE.json) identify the bundled sources.

The modified Faiss source is the raw-MinHash Jaccard engine, native source
revision `a76c9dcf4e015b199e3d4d667f47d15c513899e7`. It includes the original
MIT license and third-party notices in `vendor/faiss/`. Raw MinHash packing,
native distance arithmetic and the preceding SIMD batch filter are unchanged.

Bundled documentation, comments and notebook display outputs have been
shortened. Executable expressions and legal notices are preserved.
[SOURCE_NOTE_CHANGES.json](SOURCE_NOTE_CHANGES.json) records the original and
distributed hashes for each affected file. The revision identifiers above
refer to the source snapshots before this cleanup.

The portable JSONL streaming entry point, setup script and short runbook are new.
They screen all inputs in batches. The public configuration uses M96,
corpus/query efConstruction384, efSearch400, top-k4 and distance cutoff30.

MurmurHash3 retains its public-domain notice in `licenses/`. The custom
research SIMD and encoder files retain their original source headers; no
additional third-party license is inferred for them. See [NOTICE.md](NOTICE.md)
and the preserved native notices.
