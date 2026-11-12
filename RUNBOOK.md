# Run Milvus

Follow the steps in order to build the native engine and run the included sample.

Use an x86-64 Ubuntu 24.04 VM with sudo, at least 30 available logical CPUs,
64 GB RAM and 100 GB free disk for the build. Large datasets and results need
more disk. The CPU must support AVX-512F/DQ/VL/BW, AVX2 and POPCNT.

See [MACHINE_SPECS.md](MACHINE_SPECS.md) for small-test and larger-run VM sizes.

For reproduction, use the [reference GCP setup](MACHINE_SPECS.md#reference-gcp-setup):
AMD Genoa, 32 CPU cores, 480 GB RAM and 30 application workers.

## 1. Get the code

```bash
sudo apt-get update
sudo apt-get install -y git tmux python3.12
git clone --branch milvus --single-branch https://github.com/Kibnelson/rad_tyg1.git rad_tyg1-milvus
cd rad_tyg1-milvus
python3 scripts/check_machine.py --system milvus
```

Run the remaining commands from this directory.

`JOBS` limits compilation only; run commands below use 30 workers.

## 2. Build Milvus

On a remote VM, start a session with `tmux new -As milvus-public`, then run the
build inside it. The first build downloads many dependencies and can take hours.

```bash
PYTHON_BIN=python3.12 JOBS=2 bash scripts/setup.sh
```

Setup installs Python 3.12 and its build packages, the Python client, and Docker
with Compose if Docker is absent. An existing Docker installation must provide
Compose. Python 3.10–3.12 can be selected with `PYTHON_BIN` when its packages are
available on the VM. `JOBS=2` limits parallel compilation.

## 3. Run the included sample

No download or AWS account is needed for this 320-document test.

```bash
bash scripts/services.sh up
.venv/bin/python scripts/run.py --input data/sample.jsonl \
  --output results/sample --threads 30 --batch-size 100 --partition-capacity 1000
cat results/sample/SUMMARY.json
```

A successful run reports `status: completed` and `screened: 320`. The inserted,
index-search drop and in-batch drop counts must add up to 320. See
[README.md](README.md#saved-files) for the output files and timing scope.

## 4. Download and run more data

This example downloads 100,000 full-text FineWeb documents, then processes the
file it downloaded:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
bash scripts/services.sh up
.venv/bin/python scripts/run.py --input data/fineweb_download.jsonl \
  --output results/fineweb --threads 30 --batch-size 10000 --partition-capacity 1000000
cat results/fineweb/SUMMARY.json
```

For Common Crawl or the existing S3 datasets, follow [DATASETS.md](DATASETS.md)
and replace `--input` with the file you downloaded. `--rows` controls how many
documents to download; `--max-documents 10000` limits a run without shortening
the documents. The runtime uses 30 workers, so at least 30 logical CPUs must
be available to the process.

Use a new download filename and a new results directory when repeating a test.
The runner starts fresh and does not resume an interrupted run.

## 5. Stop the supporting services

The runner stops its own Milvus processes. After all tests, stop etcd and MinIO:

```bash
bash scripts/services.sh down
```

This keeps their local data. Run `bash scripts/services.sh up` before the next
test.
