# Installation checks

These are functional checks on the included 320-row fixture, not benchmark
throughput or recall results. Builds used Linux x86-64 and Python 3.10.

| Branch | Checks completed |
|---|---|
| [rad](https://github.com/Kibnelson/rad_tyg1/blob/rad/VALIDATION.json) | Fresh native build; scalar-distance fixture; counts, partition rollover and checkpoint reload |
| [simd_dpk](https://github.com/Kibnelson/rad_tyg1/blob/simd_dpk/VALIDATION.md) | Fresh native build; full-reference fixture; counts and one decision per input |
| [faiss_jaccard](https://github.com/Kibnelson/rad_tyg1/blob/faiss_jaccard/VALIDATION.md) | Fresh native build; raw-MinHash distances, search, save/reload and top40 fixture |
| [milvus](https://github.com/Kibnelson/rad_tyg1/blob/milvus/VALIDATION.md) | Source integrity, SDK, toolchains, SIMD and healthy etcd/MinIO; **full custom-server build and streaming fixture pending** |

The validation CPU lacks AVX-512 VPOPCNTDQ, so RAD's `cache_simd` kernel was
not executed. Its documented scalar test uses the same partition/routing path.
Milvus's fresh build requires many native dependency builds; its partial checks
must not be treated as a completed server test.

The shared fixture hashes and bounded full-text downloads from S3, FineWeb and
Common Crawl were also checked. Follow each branch's runbook for its CPU and
build requirements, and see its validation file for the precise test scope.
