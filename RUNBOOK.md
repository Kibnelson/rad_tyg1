# Run RAD

This runbook has been tested. Follow the steps in order.

These steps use Ubuntu 24.04, Python 3.12 and an x86-64 CPU with **AVX2 and
POPCNT**. Run the commands in order. The examples use **30 worker threads**;
your VM must expose at least 30 available logical CPUs.

See [MACHINE_SPECS.md](MACHINE_SPECS.md) for small-test and larger-run VM sizes.

For reproduction, use the [GCP experiment machine](MACHINE_SPECS.md#gcp-machine-used-for-the-experiments):
AMD Genoa, 32 CPU cores, 480 GB RAM and 30 application workers.

## 1. Install and build

```bash
sudo apt-get update
sudo apt-get install -y git tmux build-essential cmake ninja-build libopenblas-dev python3.12 python3.12-dev python3.12-venv
git clone --single-branch --branch rad https://github.com/Kibnelson/rad_tyg1.git rad_tyg1-rad
cd rad_tyg1-rad
python3 scripts/check_machine.py --system rad --distance-mode cache_scalar
PYTHON=python3.12 JOBS=4 bash scripts/setup.sh
```

Setup builds the native code included in this branch. It needs internet access
for build dependencies. Python 3.10–3.12 is supported; for another version,
install its matching development/venv packages and change `PYTHON` above.

`JOBS` limits compilation only; run commands below use 30 workers.

## 2. Test the included sample

```bash
.venv/bin/python scripts/check_fixture.py
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100 --partition-capacity 1000 --distance-mode cache_scalar \
  --ef-construction 128 --ef-search 150 --top-k 4 --cutoff 30
cat results/sample/summary.json
```

The summary should show `status: completed` and `screened: 320`.
`inserted`, `index_search_drops` and `in_batch_dedup_drops` should add up to 320.
The sample is already in Git; no download or AWS access is needed.
For an additional partition rollover and restart check, run
`bash scripts/smoke_test.sh`; it should print `NATIVE RAD SMOKE PASSED`.

## 3. Download and run more data

This example downloads 100,000 FineWeb documents and runs the same scalar
kernel as the sample. Stay in the cloned folder:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
.venv/bin/python scripts/run.py --input data/fineweb_download.jsonl --output results/fineweb \
  --threads 30 --batch-size 10000 --distance-mode cache_scalar \
  --ef-construction 128 --ef-search 150 --top-k 4 --cutoff 30
cat results/fineweb/summary.json
```

[DATASETS.md](DATASETS.md) shows Common Crawl and the existing S3 datasets.
Replace `--input` with the file you downloaded. Increase `--rows` for more
documents and choose new download/result paths. `--max-documents 10000` limits
a run to its first 10,000 documents. Text is kept in full.

Size RAM and disk for the input. The [partition memory policy](README.md#how-this-version-works)
can use all-resident indexes or a smaller working set, such as three partitions,
in a memory-budgeted variant. The supplied runner uses all-resident indexes.
The runtime uses 30 workers. The optional `cache_simd` kernel
requires AVX-512 VPOPCNTDQ. See [README.md](README.md) for settings and result files.

## 4. Repeat or resume

Use a new `--output` folder for a separate run. To continue after Ctrl+C, repeat
the same command with `--resume`; RAD checks the input hash and settings before
loading its last saved checkpoint. Changed graph/batch settings need a new run.
For a long job, start `tmux new -As rad` before launching the run.
