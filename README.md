# RAD and streaming deduplication baselines

Choose one system branch. Each branch has its own setup, a small included
fixture, a streaming runner and a short runbook.

| System | Branch | Instructions |
|---|---|---|
| RAD: bitmap HNSW with Bloom routing | [rad](https://github.com/Kibnelson/rad_tyg1/tree/rad) | [Runbook](https://github.com/Kibnelson/rad_tyg1/blob/rad/RUNBOOK.md) |
| Milvus: native MinHash-LSH | [milvus](https://github.com/Kibnelson/rad_tyg1/tree/milvus) | [Runbook](https://github.com/Kibnelson/rad_tyg1/blob/milvus/RUNBOOK.md) |
| SIMD-DPK: full-reference baseline | [simd_dpk](https://github.com/Kibnelson/rad_tyg1/tree/simd_dpk) | [Runbook](https://github.com/Kibnelson/rad_tyg1/blob/simd_dpk/RUNBOOK.md) |
| Faiss(Jaccard): raw MinHash HNSW | [faiss_jaccard](https://github.com/Kibnelson/rad_tyg1/tree/faiss_jaccard) | [Runbook](https://github.com/Kibnelson/rad_tyg1/blob/faiss_jaccard/RUNBOOK.md) |

Start with the included data; no AWS account is needed for that test:

```bash
git clone --branch rad --single-branch https://github.com/Kibnelson/rad_tyg1.git
cd rad_tyg1
# Follow RUNBOOK.md for setup and the sample command.
```

The examples target Linux x86-64 and Python 3.10–3.12. Native CPU/build
requirements differ by system; check the chosen branch before installing.
Milvus builds the modified server and SDK, not an interchangeable stock image.

[DATASETS.md](DATASETS.md) explains the included S3-derived fixture, the four
existing S3 dataset prefixes, and public Common Crawl/FineWeb downloads.
Real-data runs screen every document in batches, starting from an empty index.
They do not use the synthetic microbenchmark's 900K-corpus/100K-query split.

Run outputs record the selected parameters and measured counters/timings.
The included fixture validates installation and stream processing; it does not
reproduce the paper's large-run throughput or establish recall on real data.
[Source and data notices](NOTICE.md) retain upstream attribution.
