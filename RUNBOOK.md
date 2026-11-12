# Test and run

These commands target **Ubuntu 24.04 with Python 3.12**. Use Linux x86-64
with **AVX2 and POPCNT**. Python 3.10–3.12 is supported.

See [MACHINE_SPECS.md](MACHINE_SPECS.md) for small-test and larger-run VM sizes.

## 1. Install and build

```bash
sudo apt-get update
sudo apt-get install -y git tmux python3.12 python3.12-venv python3.12-dev g++
git clone --single-branch --branch simd_dpk https://github.com/Kibnelson/rad_tyg1.git rad_tyg1-simd_dpk
cd rad_tyg1-simd_dpk
python3 scripts/check_machine.py --system simd_dpk
PYTHON=python3.12 bash setup.sh
```

Setup compiles the SIMD modules and runs native duplicate checks.

## 2. Test the included data

No download or AWS account is needed:

```bash
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 30 --batch-size 100
cat results/sample/summary.json
```

Check that the run completes and that inserted and dropped documents account
for every input row. Use a VM with at least 30 available logical CPUs.

## 3. Download and test more data

This downloads 100,000 full FineWeb documents, keeping their order:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
.venv/bin/python scripts/run.py --input data/fineweb_download.jsonl \
  --output results/fineweb --threads 30 --batch-size 10000
cat results/fineweb/summary.json
```

Increase `--rows` to download more documents. For S3 or Common Crawl, follow
[DATASETS.md](DATASETS.md) and replace `--input` with the downloaded file's path.
Use new download filenames and output directories when repeating a test.
Input documents keep their full text; all are checked before insertion.

## 4. Run longer jobs

Use `tmux new -As simd-dpk`, then run the command inside it. `nproc` shows your
VM's available CPUs; these commands require at least 30. Each run starts fresh
and cannot resume.
The reference grows on disk; allow space for retained signatures and a temporary
uint64 reference file. Large runs can be slow.

The output includes `summary.json`, `batches.csv`, accepted IDs and a decision
for each input document. Keep these files together when inspecting a run.
