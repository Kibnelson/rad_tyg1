# Test and run

Use Linux x86-64 with **AVX-512F/CD/DQ/BW/VL**, AVX2, F16C, FMA, POPCNT
and BMI2 (for example, Intel Skylake Xeon or newer), Python **3.10–3.12**,
CMake ≥3.24, Ninja and a C++20 compiler. Setup builds the vendored modified
Faiss; installing the ordinary `faiss-cpu` wheel is insufficient.

```bash
git clone --single-branch --branch faiss_jaccard https://github.com/Kibnelson/rad_tyg1.git
cd rad_tyg1
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev g++ cmake ninja-build libopenblas-dev
bash setup.sh
.venv/bin/python scripts/run.py --input data/sample.jsonl --output results/sample \
  --threads 4 --batch-size 100 --partition-capacity 1000
cat results/sample/summary.json
```

For a selected interpreter/build size: `PYTHON=python3.12 JOBS=4 bash setup.sh`.
Setup checks the actual native distance, identity search and save/reload.

Download more data using [DATASETS.md](DATASETS.md), then run, for example:

```bash
.venv/bin/python scripts/run.py --input data/cc_main_download.jsonl \
  --output results/cc_main --threads 28 --batch-size 10000 \
  --partition-capacity 30000000
```

Here `--partition-capacity` is a retained-vector ceiling for **one graph**.
Choose a ceiling and VM memory large enough for the input; the baseline does
not create RAD partitions. Tune `--M`, `--ef-construction`, `--ef-search`,
`--top-k` and `--cutoff` explicitly when comparing configurations.

Output must be a new directory; this compact driver starts fresh and does not
resume runs. For long runs, use `tmux new -As faiss-jaccard` and launch inside it.
Processing QPS includes input, encoding, in-batch filtering, search, decisions
and insertion; evidence/final graph writes are outside that timer. Distance
≤30 is a raw-MinHash estimate and does not guarantee exact Jaccard ≥0.70.
Top-k evidence grows with screened rows × top-k; allow disk space for long runs.
