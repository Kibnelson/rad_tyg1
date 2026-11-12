# Source provenance

[milvus_sources.lock.json](milvus_sources.lock.json) identifies the bundled
native source files. `scripts/check_milvus_sources.py` verifies their hashes
before build and launch. The source trees and public upstream revisions are
included in this branch.

| Component | Public upstream revision | License |
|---|---|---|
| Modified Milvus v3.0.5 | `af15c807513eb6508ae22d0f3a6484e8937be1a5` | [Apache-2.0](external/milvus/LICENSE) |
| Modified Knowhere v3.0.17 | `db9598469caba6b2d9b85406ecda3b3c6edd5d9a` | [Apache-2.0](external/knowhere/LICENSE) |
| Modified pymilvus v3.0.2 | `5d478b094f3c37cceb3b3ba157ce3d6aab55ea11` | [Apache-2.0](external/pymilvus/LICENSE) |
| Milvus protocol definitions | `9e4f0ebc92af3ecf9e13c12350c6fce13d0893fa` | Included upstream notices |

Public upstreams: [Milvus](https://github.com/milvus-io/milvus),
[Knowhere](https://github.com/zilliztech/knowhere),
[pymilvus](https://github.com/milvus-io/pymilvus) and
[milvus-proto](https://github.com/milvus-io/milvus-proto). Their notices and
licenses remain in the source trees. MurmurHash3's public-domain notice is
preserved in `external/simd/MurmurHash3.*`.

The JSONL runner uses the included encoder, SIMD batch filter, native store,
capacity/sealing adapter and SDK. Store and search entry points use top-k4.
The local configuration lowers the index-building minimum from 1,024 rows to
one so small test segments receive a native LSH index.

Setup pins Go 1.26.9, Rust 1.92.0 plus the included Tantivy target's Rust 1.89
toolchain, Conan 2.25.1 and the source SDK at version 3.0.2. Go and rustup
downloads are checked against checksums from their official distribution
services. Python dependency pins are in [requirements.txt](requirements.txt).
Dependency downloads require network access.
