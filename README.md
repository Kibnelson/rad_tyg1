# RAD: Online Fuzzy Deduplication for Large-Scale Evolving Datasets

This repository contains RAD and three comparison systems for removing
near-duplicate text from a document stream. Each branch includes native source
code, an installation runbook, a small test dataset and commands for larger inputs.

RAD checks each incoming batch against documents already kept. It uses MinHash
bitmap signatures, Bloom routing and a partitioned HNSW index. Accepted documents
fill the active partition in order; a full partition is sealed and a new one opens.
Queries search the partitions selected by the Bloom filters.

## Choose a system

Open a runbook and follow its commands in order. It clones the correct branch
into a separate folder; you do not need to clone `main` first.

| System | Branch | Start here |
|---|---|---|
| RAD | [rad](https://github.com/Kibnelson/rad_tyg1/tree/rad) | [RAD runbook](https://github.com/Kibnelson/rad_tyg1/blob/rad/RUNBOOK.md) |
| Milvus | [milvus](https://github.com/Kibnelson/rad_tyg1/tree/milvus) | [Milvus runbook](https://github.com/Kibnelson/rad_tyg1/blob/milvus/RUNBOOK.md) |
| SIMD-DPK | [simd_dpk](https://github.com/Kibnelson/rad_tyg1/tree/simd_dpk) | [SIMD-DPK runbook](https://github.com/Kibnelson/rad_tyg1/blob/simd_dpk/RUNBOOK.md) |
| Faiss(Jaccard) | [faiss_jaccard](https://github.com/Kibnelson/rad_tyg1/tree/faiss_jaccard) | [Faiss(Jaccard) runbook](https://github.com/Kibnelson/rad_tyg1/blob/faiss_jaccard/RUNBOOK.md) |

The `main` branch holds shared documentation, sample data and download tools.
RAD and Faiss(Jaccard) use **M96, efConstruction 128, efSearch 150 and top-k 4**.
RAD applies efConstruction 128 to corpus and accepted-query insertions. Its
nominal bitmap-score threshold is **0.7**, implemented as the quantized
bitmap-distance rule **D ≤ 30**; this is not an exact text-Jaccard test.
Milvus uses its native MinHash-LSH index with top-k 4. SIMD-DPK checks the full
retained reference and has no HNSW or ranked top-k setting.

## Quick start: RAD on the included sample

Use Ubuntu 24.04, Python 3.12 and an x86-64 VM with AVX2, POPCNT and at least
**30 available logical CPUs**. Commands use **30 worker threads**; a 32-vCPU VM
is recommended. See [MACHINE_SPECS.md](MACHINE_SPECS.md) for memory, disk and
the additional CPU instructions needed by each system.

```bash
sudo apt-get update
sudo apt-get install -y git tmux build-essential cmake ninja-build libopenblas-dev python3.12 python3.12-dev python3.12-venv
git clone --single-branch --branch rad https://github.com/Kibnelson/rad_tyg1.git rad_tyg1-rad
cd rad_tyg1-rad
python3 scripts/check_machine.py --system rad --distance-mode cache_scalar
PYTHON=python3.12 JOBS=4 bash scripts/setup.sh
.venv/bin/python scripts/check_fixture.py
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100 --partition-capacity 1000 \
  --distance-mode cache_scalar --ef-construction 128 --ef-search 150 --top-k 4 --cutoff 30
cat results/sample/summary.json
```

The sample needs no account or download. Check that the run completes and that
inserted and dropped documents account for every input row. Use a new output
folder for another test. The [RAD runbook](https://github.com/Kibnelson/rad_tyg1/blob/rad/RUNBOOK.md)
also covers larger inputs and checkpoint resume.

## Data

The included sample is `data/sample.jsonl`. [DATASETS.md](DATASETS.md) shows how
to download full-text FineWeb or Common Crawl documents without an AWS account,
or use the saved S3 datasets if you already have bucket access.

For example, after completing a system's setup:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
```

Return to that system's runbook and use the downloaded file as `--input`.
Downloads keep full document text and source order. To compare systems, download
once and pass the same file to every runner. Keep its `.provenance.json` file,
which records the source and file hash.

## Code and documentation

| Component | RAD branch |
|---|---|
| Streaming entry point | [scripts/run.py](https://github.com/Kibnelson/rad_tyg1/blob/rad/scripts/run.py) |
| Partition storage | [partitions.py](https://github.com/Kibnelson/rad_tyg1/blob/rad/src/radv2/partitions.py) |
| Bloom routing and partition search | [routing.py](https://github.com/Kibnelson/rad_tyg1/blob/rad/src/radv2/routing.py) |
| MinHash and bitmap encoding | [encoding.py](https://github.com/Kibnelson/rad_tyg1/blob/rad/src/radv2/encoding.py) |
| Native bitmap helpers | [bitmap_native.py](https://github.com/Kibnelson/rad_tyg1/blob/rad/src/radv2/bitmap_native.py) |

- [RUNBOOK.md](RUNBOOK.md): links to each system's installation steps.
- [DATASETS.md](DATASETS.md): sample data and download commands.
- [MACHINE_SPECS.md](MACHINE_SPECS.md): machine sizes and CPU requirements.
- [VALIDATION.md](VALIDATION.md): completed installation checks and remaining work.
- [NOTICE.md](NOTICE.md): source and data attribution.
