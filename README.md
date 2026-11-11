# RAD

Streaming MinHash-bitmap HNSW with sequential 1M-retained-document partitions
and per-partition Bloom routing. This branch includes its modified native
Faiss and in-batch SIMD sources; it does not fetch a private repository.

## Test the included data

Linux x86-64, AVX2 + POPCNT, Python 3.10–3.12:

```bash
git clone --single-branch --branch rad https://github.com/Kibnelson/rad_tyg1.git
cd rad_tyg1
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build swig libopenblas-dev python3-dev python3-venv
bash scripts/setup.sh
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 4 --batch-size 100 --partition-capacity 1000 --distance-mode cache_scalar
cat results/sample/summary.json
bash scripts/smoke_test.sh
```

The sample command uses the scalar distance kernel so it also runs on AVX2-only
CPUs. `cache_simd` is the benchmark kernel and requires AVX-512 VPOPCNTDQ.
The selected kernel is recorded; do not compare their timings as identical builds.

See [RUNBOOK.md](RUNBOOK.md) for downloaded inputs, a larger run and resume.
See [DATASETS.md](DATASETS.md) for the fixture's provenance and larger inputs.

Functional installation checks: [VALIDATION.json](VALIDATION.json).
