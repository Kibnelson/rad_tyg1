# Installation checks

These notes record installation and functional checks on the included sample.
The runbooks use 30 worker threads; local checks use the resources available
on the validation host.

| Branch | Checks completed |
|---|---|
| [rad](https://github.com/Kibnelson/rad_tyg1/blob/rad/VALIDATION.json) | Fresh native build; scalar-distance fixture; counts, partition rollover and checkpoint reload |
| [simd_dpk](https://github.com/Kibnelson/rad_tyg1/blob/simd_dpk/VALIDATION.md) | Fresh native build; full-reference fixture; counts and one decision per input |
| [faiss_jaccard](https://github.com/Kibnelson/rad_tyg1/blob/faiss_jaccard/VALIDATION.md) | Native build; raw-MinHash distances, top-k4 search and save/reload |
| [milvus](https://github.com/Kibnelson/rad_tyg1/blob/milvus/VALIDATION.md) | Source integrity, SDK, toolchains, SIMD and healthy etcd/MinIO |

The validation CPU lacks AVX-512 VPOPCNTDQ, so RAD's `cache_simd` kernel was
not executed. Its documented scalar test uses the same partition/routing path.

The shared fixture hashes and bounded full-text downloads from S3, FineWeb and
Common Crawl were also checked. Follow each branch's runbook for its CPU and
build requirements, and see its validation file for the precise test scope.
