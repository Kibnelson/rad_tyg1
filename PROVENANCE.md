# Source provenance

The native source inventory is pinned by `milvus_sources.lock.json`. `scripts/check_milvus_sources.py` verifies every listed native file before build and launch. No private Git checkout, cloud account, VM path or credentials are required to build the engine.

| Component | Public upstream revision | License |
|---|---|---|
| Milvus v3.0.5 plus experiment modifications | `af15c807513eb6508ae22d0f3a6484e8937be1a5` | [Apache-2.0](external/milvus/LICENSE) |
| Knowhere v3.0.17 plus experiment modifications | `db9598469caba6b2d9b85406ecda3b3c6edd5d9a` | [Apache-2.0](external/knowhere/LICENSE) |
| pymilvus v3.0.2 plus experiment modifications | `5d478b094f3c37cceb3b3ba157ce3d6aab55ea11` | [Apache-2.0](external/pymilvus/LICENSE) |
| Milvus protocol definitions | `9e4f0ebc92af3ecf9e13c12350c6fce13d0893fa` | Included upstream notices |

Public upstreams: [Milvus](https://github.com/milvus-io/milvus), [Knowhere](https://github.com/zilliztech/knowhere), [pymilvus](https://github.com/milvus-io/pymilvus), [milvus-proto](https://github.com/milvus-io/milvus-proto). Their notices and license files remain in their source trees. MurmurHash3's public-domain notice remains in `external/simd/MurmurHash3.*`. No new repository-wide license is asserted for the experiment code.

The experiment source snapshot corresponds to native base `96b300fc4044fd9edcc2f67d7af9a4367fbac2f4`, runtime `1f35a23e02c61ce80eb0e70b45e2f1b48a31ada0`, and full-text tools `7a40e0692339bccf9eb2488dbe56918dcad83241`. These record provenance, not download requirements. The portable runner uses the original encoder, SIMD implementation, native store, capacity/sealing adapter and SDK. Unused private dataset/publication/recovery wrappers were removed. The local config lowers the index-building minimum from 1,024 rows to one so tiny test segments receive a native LSH index. Its JSONL input and output/report wrapper are new; this package does not claim to reproduce prior scores on the toy data.

The setup pins Go 1.26.9, Rust 1.92.0 plus the included Tantivy target's Rust 1.89 toolchain, Conan 2.25.1 and the source SDK at version 3.0.2. Go and rustup downloads are checked against checksums from their official distribution services. Python dependency pins are in `requirements.txt`; transitive packages and native upstream dependency downloads still require network access.
