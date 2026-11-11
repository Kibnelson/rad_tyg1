# RAD and streaming deduplication baselines

Start with one system below. Its runbook takes you from installing the tools
to running a small test, then downloading and processing more documents.
Each system has its own branch and clone folder, so you can try several systems.
Run commands use **30 worker threads** on a VM with at least **30 available
logical CPUs**; a 32-vCPU VM is recommended.

| System | Branch | Start here |
|---|---|---|
| RAD | [rad](https://github.com/Kibnelson/rad_tyg1/tree/rad) | [RAD runbook](https://github.com/Kibnelson/rad_tyg1/blob/rad/RUNBOOK.md) |
| Milvus | [milvus](https://github.com/Kibnelson/rad_tyg1/tree/milvus) | [Milvus runbook](https://github.com/Kibnelson/rad_tyg1/blob/milvus/RUNBOOK.md) |
| SIMD-DPK | [simd_dpk](https://github.com/Kibnelson/rad_tyg1/tree/simd_dpk) | [SIMD-DPK runbook](https://github.com/Kibnelson/rad_tyg1/blob/simd_dpk/RUNBOOK.md) |
| Faiss(Jaccard) | [faiss_jaccard](https://github.com/Kibnelson/rad_tyg1/tree/faiss_jaccard) | [Faiss(Jaccard) runbook](https://github.com/Kibnelson/rad_tyg1/blob/faiss_jaccard/RUNBOOK.md) |

[MACHINE_SPECS.md](MACHINE_SPECS.md) lists VM sizes and CPU requirements.
The runbooks use Ubuntu 24.04 and Python 3.12. They list each system's CPU,
memory and build requirements before the commands. Native source code is
included; Milvus builds the custom server and SDK used by this project.

The 320-document sample is already in Git and needs no AWS account.
[DATASETS.md](DATASETS.md) explains how to download public FineWeb/Common Crawl
text or use the existing S3 datasets if you have bucket access. Larger downloads
keep full document text and order. Every input document is checked before it
can be inserted.

The small sample checks installation and insert/drop handling. Its timings
are not paper benchmark results, and real-data recall is not measured.
[VALIDATION.md](VALIDATION.md) records the checks completed so far;
Milvus's full custom-server build and streaming test are still pending.
[NOTICE.md](NOTICE.md) records source and data attribution.
