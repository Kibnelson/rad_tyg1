# Test and run

Use Linux x86-64 with **AVX2 and POPCNT**, Python **3.10–3.12**, and a C++ compiler.

```bash
git clone --single-branch --branch simd_dpk https://github.com/Kibnelson/rad_tyg1.git
cd rad_tyg1
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev g++
bash setup.sh
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 4 --batch-size 100
cat results/sample/summary.json
```

If the default Python is newer, use `PYTHON=python3.12 bash setup.sh`.
Setup compiles the actual SIMD modules and runs native duplicate checks.

For a larger run, first download data using [DATASETS.md](DATASETS.md), then:

```bash
.venv/bin/python scripts/run.py --input data/cc_main_download.jsonl \
  --output results/cc_main --threads 28 --batch-size 10000
```

Choose threads within the VM's CPU allocation; the input text and order stay
unchanged. All source rows follow search/admission before insertion. Output must
be a new directory. This compact driver starts fresh; it does not resume runs.
Use tmux for long jobs: `tmux new -As simd-dpk`, then run the command inside it.

Processing QPS is screened rows / summed batch processing seconds, including
reference staging and native rebuilding. Evidence files and final reporting
are outside this timer. The reference grows on disk; ensure free disk for both
retained signatures and a uint64 union staging file. Large runs can be slow.
