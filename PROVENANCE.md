# Source provenance

Research snapshot: `faada9ad388b5c0f1c5e4d13052a2a1ebab247b0`.
[Exact copied-file SHA256 values](PROVENANCE.json) bind the original sources.

The two native SIMD modules and MinHash encoder are unchanged. The
full-reference observer reports query-related edges without changing the
original deduplication algorithm. This branch does not vendor IBM DPK or
launch an IBM/Ray ground-truth job; SIMD-DPK is the comparison label.

The portable JSONL streaming entry point, setup script and short runbook are new.
They screen all inputs in batches, rather than replaying the older prepared
900K-corpus/100K-query workload. No historical figure numbers are implied.

MurmurHash3 retains its public-domain notice in `licenses/`. The custom
research SIMD and encoder files retain their original source headers; no
additional third-party license is inferred for them. See [NOTICE.md](NOTICE.md)
and the preserved native notices.
