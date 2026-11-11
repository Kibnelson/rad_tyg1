# Source provenance

Research snapshot: `a4b78aaee37dea08662c36d97170f0e45cdbce7a`.
[Exact copied-file SHA256 values](PROVENANCE.json) bind the original sources.

The modified Faiss source is the raw-MinHash Jaccard engine, native source
revision `a76c9dcf4e015b199e3d4d667f47d15c513899e7`. It includes the original
MIT license and third-party notices in `vendor/faiss/`. Raw MinHash packing,
native distance arithmetic and the preceding SIMD batch filter are unchanged.

The portable JSONL streaming entry point, setup script and short runbook are new.
They screen all inputs in batches, rather than replaying the older prepared
900K-corpus/100K-query workload. No historical figure numbers are implied.

MurmurHash3 retains its public-domain notice in `licenses/`. The custom
research SIMD and encoder files retain their original source headers; no
additional third-party license is inferred for them. See [NOTICE.md](NOTICE.md)
and the preserved native notices.
