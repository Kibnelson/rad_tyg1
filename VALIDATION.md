# Installation checks

The Milvus native engine and MinHash-LSH configuration were tested on GCP using
the experiment runbooks. The public runbook uses the same engine and index settings.

The record below separates that GCP check from the public-package checks on
Linux x86-64 with Python 3.10:

| Check | Result |
|---|---|
| Milvus native engine and MinHash-LSH profile using the experiment runbooks on GCP | Passed |
| SHA-256 inventory of pinned native source files | Passed |
| Source SDK wheel build, installation and import as pymilvus 3.0.2 | Passed |
| Pinned Go/Rust toolchain installation with official checksums | Passed |
| Minimal Ubuntu 24.04 Python 3.12 bootstrap, matching development headers, new venv and pip | Passed |
| Original SIMD compilation and repeat-filtering smoke test | Passed |
| Public adapter imports, top4 retained profile construction, candidate merge and checksum helpers | Passed |
| Python/shell syntax and Docker Compose configuration | Passed |
| Pinned etcd and MinIO services start and report healthy | Passed |

The public tiny-input profile explicitly lowers `indexCoord.segment.minSegmentNumRowsToEnableIndex` from 1,024 to one, allowing its final small segment to build a real native LSH index. The 112-value signatures, 14×8 LSH, top4 admission, and retained-row partitioning are preserved.
