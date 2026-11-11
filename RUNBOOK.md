# Test and run

These commands target **Ubuntu 24.04 with Python 3.12**. Use Linux x86-64
with **AVX-512F/CD/DQ/BW/VL**, AVX2, F16C, FMA, POPCNT and BMI2. Python
3.10–3.12, CMake ≥3.24, Ninja and a C++20 compiler are supported.

See [MACHINE_SPECS.md](MACHINE_SPECS.md) for small-test and larger-run VM sizes.

## 1. Install and build

```bash
sudo apt-get update
sudo apt-get install -y git tmux python3.12 python3.12-venv python3.12-dev g++ cmake ninja-build libopenblas-dev
git clone --single-branch --branch faiss_jaccard https://github.com/Kibnelson/rad_tyg1.git rad_tyg1-faiss_jaccard
cd rad_tyg1-faiss_jaccard
python3 scripts/check_machine.py --system faiss_jaccard
PYTHON=python3.12 JOBS=4 bash setup.sh
```

Setup builds the included modified Faiss and checks native distance, identity
search and save/reload. The ordinary `faiss-cpu` package cannot run this engine.

## 2. Test the included data

No download or AWS account is needed:

```bash
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 4 --batch-size 100 --partition-capacity 1000
cat results/sample/summary.json
```

On success, `screened` is **320** and `inserted` + `index_search_drops` +
`in_batch_dedup_drops` totals 320. Individual counts can vary because HNSW is
approximate. These are installation checks, not benchmark scores. Use fewer
threads if your VM has fewer than four available CPUs.

## 3. Download and test more data

This downloads 100,000 full FineWeb documents, keeping their order:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
.venv/bin/python scripts/run.py --input data/fineweb_download.jsonl \
  --output results/fineweb --threads 4 --batch-size 10000 --partition-capacity 100000
cat results/fineweb/summary.json
```

`--rows` is the number of documents to download. `--partition-capacity` limits
documents kept in **one graph**; it does not create RAD partitions. Increase
this ceiling, and allow enough memory, when downloading more documents. For S3
or Common Crawl, follow [DATASETS.md](DATASETS.md) and replace `--input` with
the downloaded file's path. Use new download filenames and output directories
when repeating a test. Input documents keep their full text.

## 4. Run longer jobs

Use `tmux new -As faiss-jaccard`, then run the command inside it. `nproc` shows
your VM's available CPUs; keep `--threads` within that count. Each run starts
fresh and cannot resume. Tune `--M`, `--ef-construction`, `--ef-search`, `--top-k` and
`--cutoff` explicitly for another configuration.

Processing QPS includes input, encoding, in-batch filtering, search, decisions
and insertion. Evidence and final graph writes are outside this timer.
Distance ≤30 compares MinHash signatures; it does not guarantee exact Jaccard
≥0.70. By default, the runner saves all ordered top40 candidates before
admission, so allow disk space as the number of documents grows.
