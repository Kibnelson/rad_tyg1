# Test and run

These commands target **Ubuntu 24.04 with Python 3.12**. Use Linux x86-64
with **AVX-512F/CD/DQ/BW/VL**, AVX2, F16C, FMA, POPCNT and BMI2. Python
3.10–3.12, CMake ≥3.24, Ninja and a C++20 compiler are supported.

See [MACHINE_SPECS.md](MACHINE_SPECS.md) for small-test and larger-run VM sizes.

For reproduction, use the [GCP experiment machine](MACHINE_SPECS.md#gcp-machine-used-for-the-experiments):
AMD Genoa, 32 CPU cores, 480 GB RAM and 30 application workers.

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

`JOBS` limits compilation only; run commands below use 30 workers.

## 2. Test the included data

No download or AWS account is needed:

```bash
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100 --partition-capacity 1000 \
  --ef-construction 384 --ef-search 400 --top-k 4
cat results/sample/summary.json
```

On success, `screened` is **320** and `inserted` + `index_search_drops` +
`in_batch_dedup_drops` totals 320. Individual counts can vary because HNSW is
approximate. Use a VM
with at least 30 available logical CPUs for these commands.

## 3. Download and test more data

This downloads 100,000 full FineWeb documents, keeping their order:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
.venv/bin/python scripts/run.py --input data/fineweb_download.jsonl \
  --output results/fineweb --threads 30 --batch-size 10000 --partition-capacity 100000 \
  --ef-construction 384 --ef-search 400 --top-k 4
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
your VM's available CPUs; these commands require at least 30. Each run starts
fresh and cannot resume. The public preset fixes corpus/query efConstruction
at 384, efSearch at 400 and top-k at 4. Use `--M` and `--cutoff` explicitly to
change graph connectivity or the admission threshold.

Distance ≤30 compares MinHash signatures; it does not guarantee exact Jaccard
≥0.70. The runner saves all ordered top-k4 candidates before
admission, so allow disk space as the number of documents grows.
