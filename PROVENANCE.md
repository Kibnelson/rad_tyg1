# RAD source and test record

The native Faiss tree, SIMD sources, bitmap/Bloom helpers and partitioned
index modules were copied from runtime revision
`65a5023a17ec83b6745dc8488329a4f7ecb42c5f` of `Kibnelson/rad_workspacev2`.
These sources are included here; setup never clones that repository.
[Source manifest](PROVENANCE.json) records the distribution scope.

The new public runner adds JSONL input, portable paths, per-batch evidence,
timing reports and local checkpoint resume. It makes no separate warmup
searches and does not claim to reproduce prior figure throughput.
The original modified native indexing, bitmap distance, in-batch SIMD and
Bloom-routing implementations are retained.

[VALIDATION.json](VALIDATION.json) records a fresh native-source build and
tests on the included fixture, including partition rollover, raw evidence,
checkpoint reload and bitmap parity. These are functional tests, not benchmark
measurements. They used native `cache_scalar` because the validation CPU lacks
AVX-512 VPOPCNTDQ; unsupported `cache_simd` is refused explicitly.

Faiss's MIT license and third-party notices are preserved in
`external/faiss/LICENSE` and `external/faiss/THIRD_PARTY_NOTICES`.
MurmurHash3 retains its public-domain notice in `external/simd/MurmurHash3.h`.
