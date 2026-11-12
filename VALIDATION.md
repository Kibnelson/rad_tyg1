# Validation status

The portable package has passed these checks on Linux x86-64 with Python 3.10:

| Check | Result |
|---|---|
| SHA-256 inventory of pinned native source files | Passed |
| Source SDK wheel build, installation and import as pymilvus 3.0.2 | Passed |
| Pinned Go/Rust toolchain installation with official checksums | Passed |
| Minimal Ubuntu 24.04 Python 3.12 bootstrap, matching development headers, new venv and pip | Passed |
| Original SIMD compilation and repeat-filtering smoke test | Passed |
| Public adapter imports, top4 retained profile construction, candidate merge and checksum helpers | Passed |
| Python/shell syntax and Docker Compose configuration | Passed |
| Pinned etcd and MinIO services start and report healthy | Passed |
| Complete custom Milvus native build and end-to-end streaming fixture | **Pending** |

The last check must not be inferred from the adapter/SIMD tests. A fresh native build was started, but the public JSONL wrapper has not yet completed its native server smoke test.

The public tiny-input profile explicitly lowers `indexCoord.segment.minSegmentNumRowsToEnableIndex` from 1,024 to one, allowing its final small segment to build a real native LSH index. The 112-value signatures, 14×8 LSH, top4 admission, and retained-row partitioning are preserved.
